//! PTY proxy with virtual terminal for persistent status bar.
//!
//! When the status bar is enabled, ai-jail interposes a PTY between
//! itself and the sandbox child. The child writes to the PTY slave
//! while ai-jail owns the real terminal.
//!
//! Rendering uses a hybrid approach:
//!   - **Primary screen**: raw pass-through with a scroll region
//!     protecting the status bar. This preserves natural terminal
//!     scrollback. vt100 processes the output in parallel for
//!     resize recovery.
//!   - **Alternate screen** (vim, less, etc.): cursor-addressed
//!     diff-rendering via vt100::state_diff. No scrollback needed.
//!
//! On resize, the vt100 virtual terminal is resized first, then
//! its state is re-rendered to the real terminal, giving ai-jail
//! full control over screen recovery.
//!
//! By default output is rendered safely: terminal control queries and string
//! commands are not passed from the sandbox to the host terminal, except a
//! window title (OSC 0/1/2), which is rebuilt from printable text only. The
//! `--terminal-passthrough` opt-in restores raw TUI terminal behavior.

use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
use nix::sys::termios::{self, SetArg, Termios};
use std::os::unix::io::{AsRawFd, BorrowedFd, OwnedFd};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Stored master raw FD for async-signal-safe resize from SIGWINCH.
static MASTER_FD: AtomicI32 = AtomicI32::new(-1);
/// Whether the current proxy reserves a row for the status bar.
static STATUS_BAR_ACTIVE: AtomicBool = AtomicBool::new(false);
/// `st_rdev` of the PTY *slave* (e.g. `/dev/pts/N`). The master fd is
/// `/dev/ptmx` (major 5), whose device number differs from the slave's, so
/// the `/proc`-based SIGWINCH fallback must match descendants' `tty_nr`
/// against THIS, not against `fstat(MASTER_FD).st_rdev`. Zero when unset
/// (unit tests that don't go through `run_with_passthrough`).
static SLAVE_DEV: AtomicI64 = AtomicI64::new(0);

/// Record the slave PTY device number for the SIGWINCH /proc fallback.
pub fn set_slave_dev(slave_fd: i32) {
    let mut st: nix::libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { nix::libc::fstat(slave_fd, &mut st) } == 0 {
        SLAVE_DEV.store(st.st_rdev as i64, Ordering::SeqCst);
    }
}

/// Set by signal handler; IO loop clears screen + redraws status bar
/// BEFORE forwarding SIGWINCH to the child, preventing ghost bars.
static SIGWINCH_PENDING: AtomicBool = AtomicBool::new(false);
const RESIZE_REDRAW_DELAY: Duration = Duration::from_millis(75);

/// Post-resize byte-capture budget (env-gated SIGWINCH trace). When the
/// trace is enabled, `handle_sigwinch` sets this to a small budget so the
/// next `handle_master_read`s log the child's bytes — i.e. whether the
/// child actually repainted after the resize (SIGWINCH was delivered).
static RESIZE_CHILD_BUDGET: AtomicUsize = AtomicUsize::new(0);

/// `AI_JAIL_TRACE_SIGWINCH` diagnostic: `1` for stderr, else a file path.
/// Traces `handle_sigwinch` / `resize_pty` / `forward_sigwinch` plus the
/// child's first bytes after a resize. Always no-op unless the env var is
/// set, so it never changes normal behavior.
fn sigwinch_trace_enabled() -> bool {
    std::env::var_os("AI_JAIL_TRACE_SIGWINCH").is_some()
}

fn trace_sigwinch(msg: &str) {
    let Some(path) = std::env::var_os("AI_JAIL_TRACE_SIGWINCH") else {
        return;
    };
    use std::io::Write;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let line = format!("{ts} {msg}\n");
    if path.to_str().map(|s| s == "1").unwrap_or(false) {
        let _ = std::io::stderr().write_all(line.as_bytes());
    } else if let Ok(mut f) =
        std::fs::OpenOptions::new().append(true).create(true).open(path)
    {
        let _ = f.write_all(line.as_bytes());
    }
}

/// Mark a SIGWINCH as pending. Called from the signal handler.
pub fn set_sigwinch_pending() {
    SIGWINCH_PENDING.store(true, Ordering::SeqCst);
}

/// Read-and-clear the SIGWINCH pending flag. Test-only — production
/// callers do this implicitly via `SIGWINCH_PENDING.swap(false, ...)`
/// inside the IO loop's `handle_sigwinch` method.
#[cfg(test)]
pub(crate) fn take_sigwinch_pending_for_test() -> bool {
    SIGWINCH_PENDING.swap(false, Ordering::SeqCst)
}

/// Resize the PTY slave to match the real terminal (minus one row
/// for the status bar). Async-signal-safe: only uses ioctl + atomics.
pub fn resize_pty() {
    let master = MASTER_FD.load(Ordering::SeqCst);
    if master < 0 {
        return;
    }
    let mut ws = unsafe { std::mem::zeroed::<nix::libc::winsize>() };
    let ret = unsafe {
        nix::libc::ioctl(
            nix::libc::STDOUT_FILENO,
            nix::libc::TIOCGWINSZ,
            &mut ws,
        )
    };
    if ret != 0 || ws.ws_row == 0 || ws.ws_col == 0 {
        return;
    }
    if STATUS_BAR_ACTIVE.load(Ordering::SeqCst) {
        if ws.ws_row < 2 {
            return;
        }
        ws.ws_row -= 1;
    }
    let rows = ws.ws_row;
    unsafe {
        nix::libc::ioctl(master, nix::libc::TIOCSWINSZ, &ws);
    }
    trace_sigwinch(&format!(
        "resize_pty: read host rows={} cols={} status_bar={} -> slave rows={}",
        rows + if STATUS_BAR_ACTIVE.load(Ordering::SeqCst) { 1 } else { 0 },
        ws.ws_col,
        STATUS_BAR_ACTIVE.load(Ordering::SeqCst),
        rows
    ));
}

/// Return host-namespace process-group ids of the PTY's foreground
/// processes, resolving across `bwrap --unshare-pid`.
///
/// `TIOCGPGRP` on the master yields a *child-namespace* pgid, which is
/// unaddressable from the parent (`kill(-pgid)` -> `ESRCH`). Walk the
/// spawned sandbox process's host-namespace descendants and collect the
/// `tpgid` (field 8 of `/proc/<pid>/stat` — the foreground pgrp *as seen
/// from the host namespace*, which `kill(2)` can target despite the nested
/// PID namespace) of every descendant whose controlling terminal is this PTY.
///
/// We prefer the descendant whose `tty_nr` (field 7) equals the master
/// PTY device (`fstat(MASTER_FD).st_rdev`). If that precise match fails
/// (e.g. device-encoding quirks or a helper sharing the tty), we fall back
/// to signalling *every* tty-holding descendant, so the real shell (the
/// actual foreground tty holder) still receives SIGWINCH.
fn child_host_namespace_pgrps() -> Vec<nix::libc::pid_t> {
    let child = crate::signals::child_pid();
    if child <= 0 {
        trace_sigwinch(&format!(
            "child_host_namespace_pgrps: no child pid (child={child})"
        ));
        return Vec::new();
    }
    let master = MASTER_FD.load(Ordering::SeqCst);
    if master < 0 {
        return Vec::new();
    }
    // Prefer the recorded slave device (the child's controlling tty device),
    // which is what `/proc/<pid>/stat` tty_nr reports. Master (`/dev/ptmx`)
    // has a different st_rdev, so never match against it.
    let dev = SLAVE_DEV.load(Ordering::SeqCst);
    let mut st: nix::libc::stat = unsafe { std::mem::zeroed() };
    let master_dev: Option<i64> = if dev != 0 {
        None
    } else if unsafe { nix::libc::fstat(master, &mut st) } == 0 {
        Some(st.st_rdev as i64)
    } else {
        None
    };
    let dev = if dev != 0 { dev } else { master_dev.unwrap_or(0) };
    if dev == 0 {
        return Vec::new();
    }
    let our_pid = unsafe { nix::libc::getpid() };
    trace_sigwinch(&format!(
        "child_host_namespace_pgrps: child={child} slave_dev={dev:#x} master_dev={master_dev:?} our_pid={our_pid}"
    ));

    // Read /proc/<pid>/stat for a single pid; return (tpgid, precise) where
    // `precise` is true iff its controlling-tty device (tty_nr, field 7)
    // matches the master PTY device. `tpgid` (field 8) is the foreground
    // pgrp *as seen from the host namespace* — killable from here even when
    // the child sits in a nested PID namespace.
    let inspect = |pid: nix::libc::pid_t,
                   label: &str,
                   depth: usize|
     -> Option<(nix::libc::pid_t, bool)> {
        let s = match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            Ok(s) => s,
            Err(e) => {
                trace_sigwinch(&format!(
                    "  depth={depth} {label} pid={pid} stat_read_fail err={e}"
                ));
                return None;
            }
        };
        // /proc/<pid>/stat => "pid (comm) state ppp cgg sess tty tpgid ..."
        let rparen = s.rfind(')')?;
        let mut it = s[rparen + 2..].split_whitespace();
        let _state = it.next();
        let _ppid = it.next();
        let _pgrp = it.next();
        let _session = it.next();
        let tty_nr_s = it.next(); // field 7
        let tpgid_s = it.next(); // field 8
        let tty_nr_val = tty_nr_s.and_then(|v| v.parse::<i64>().ok());
        let tty_match = tty_nr_val.is_some_and(|v| v == dev);
        let tty_nonzero = tty_nr_val.is_some_and(|v| v != 0);
        trace_sigwinch(&format!(
            "  depth={depth} {label} pid={pid} tty_nr={tty_nr_s:?} (dev_match={tty_match} tty_nonzero={tty_nonzero}) tpgid={tpgid_s:?}"
        ));
        let tpg = tpgid_s.and_then(|v| v.parse::<nix::libc::pid_t>().ok())?;
        if tpg <= 0 {
            return None;
        }
        Some((tpg, tty_match))
    };

    let mut frontier: Vec<nix::libc::pid_t> = vec![child];
    let mut precise: Option<nix::libc::pid_t> = None;
    let mut tty_holders: Vec<nix::libc::pid_t> = Vec::new();
    for depth in 0..4 {
        let mut next: Vec<nix::libc::pid_t> = Vec::new();
        for pid in frontier {
            // (1) inspect this pid itself first — covers the recorded child
            //    *being* the tty holder (no-bwrap exec case, or bwrap without
            //    --new-session where the wrapper keeps the ctty).
            if let Some((tpg, is_precise)) = inspect(pid, "self", depth) {
                if is_precise {
                    precise = Some(tpg);
                    tty_holders.push(tpg);
                    break;
                }
                tty_holders.push(tpg);
            }
            // (2) then its children — covers bwrap where the shell is a
            //    descendant of the bwrap wrapper.
            let kids = match std::fs::read_to_string(format!(
                "/proc/{pid}/task/{pid}/children"
            )) {
                Ok(s) => s,
                Err(e) => {
                    trace_sigwinch(&format!(
                        "  depth={depth} pid={pid} children_read_fail err={e}"
                    ));
                    String::new()
                }
            };
            trace_sigwinch(&format!(
                "  depth={depth} pid={pid} children={:?}",
                kids.split_whitespace().collect::<Vec<_>>()
            ));
            for k in kids.split_whitespace() {
                let k: i32 = match k.parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if k == our_pid {
                    continue;
                }
                next.push(k);
                if let Some((tpg, is_precise)) = inspect(k, "child", depth) {
                    if is_precise {
                        precise = Some(tpg);
                        tty_holders.push(tpg);
                        break;
                    }
                    tty_holders.push(tpg);
                }
            }
            if precise.is_some() {
                break;
            }
        }
        if precise.is_some() {
            break;
        }
        frontier = next;
    }
    tty_holders.dedup();
    if let Some(p) = precise {
        trace_sigwinch(&format!(
            "child_host_namespace_pgrps: precise host_pgrp={p}; tty_holders_all={tty_holders:?}"
        ));
        vec![p]
    } else {
        trace_sigwinch(&format!(
            "child_host_namespace_pgrps: no precise match (dev={dev:#x}); signaling all tty_holders={tty_holders:?}"
        ));
        tty_holders
    }
}

/// Explicitly send SIGWINCH to the PTY foreground process group.
/// `TIOCSWINSZ` (in `resize_pty`) should do this via the kernel, but
/// bwrap's PID namespace can prevent delivery. This is the reliable
/// fallback; under a plain (no bwrap) run the `kill(-pgrp)` succeeds and
/// we never reach the fallback.
fn forward_sigwinch() {
    let master = MASTER_FD.load(Ordering::SeqCst);
    if master < 0 {
        return;
    }
    let mut pgrp: nix::libc::pid_t = 0;
    let tc = unsafe { nix::libc::ioctl(master, nix::libc::TIOCGPGRP, &mut pgrp) };
    if tc == 0 && pgrp > 0 {
        let r = unsafe { nix::libc::kill(-pgrp, nix::libc::SIGWINCH) };
        if r == 0 {
            trace_sigwinch(&format!(
                "forward_sigwinch: tcgetpgrp_ok pgrp={pgrp} kill_ok"
            ));
            return;
        }
        let e = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        trace_sigwinch(&format!(
            "forward_sigwinch: tcgetpgrp_ok pgrp={pgrp} kill_ret={r} errno={e} (child-ns pgrp, ESRCH expected under bwrap --unshare-pid)"
        ));
    } else {
        let e = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        trace_sigwinch(&format!(
            "forward_sigwinch: tcgetpgrp failed (ret={tc} pgrp={pgrp} errno={e})"
        ));
    }
    // bwrap --unshare-pid fallback: deliver SIGWINCH to the child's
    // host-namespace foreground pgrp(s) (see `child_host_namespace_pgrps`).
    for host_pgrp in child_host_namespace_pgrps() {
        let r = unsafe { nix::libc::kill(-host_pgrp, nix::libc::SIGWINCH) };
        let e = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        trace_sigwinch(&format!(
            "forward_sigwinch[fallback]: host_pgrp={host_pgrp} kill_ret={r} errno={e}"
        ));
    }
}

fn enter_raw_mode() -> Result<Termios, String> {
    let stdin = std::io::stdin();
    let saved =
        termios::tcgetattr(&stdin).map_err(|e| format!("tcgetattr: {e}"))?;
    let mut raw = saved.clone();
    termios::cfmakeraw(&mut raw);
    termios::tcsetattr(&stdin, SetArg::TCSANOW, &raw)
        .map_err(|e| format!("tcsetattr raw: {e}"))?;
    Ok(saved)
}

fn restore_mode(saved: &Termios) {
    let stdin = std::io::stdin();
    let _ = termios::tcsetattr(&stdin, SetArg::TCSANOW, saved);
}

struct RawModeGuard(Option<Termios>);

impl RawModeGuard {
    fn new(saved: Termios) -> Self {
        Self(Some(saved))
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        if let Some(saved) = self.0.take() {
            restore_mode(&saved);
        }
    }
}

fn set_initial_size(fd: &OwnedFd, rows: u16, cols: u16) {
    let ws = nix::libc::winsize {
        ws_row: rows,
        ws_col: cols,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    unsafe {
        nix::libc::ioctl(fd.as_raw_fd(), nix::libc::TIOCSWINSZ, &ws);
    }
}

/// Set terminal scroll region to rows 1..content_rows (1-based).
/// Status bar lives on row content_rows+1, outside the region.
fn set_scroll_region(fd: i32, content_rows: u16) {
    let seq = format!("\x1b[1;{content_rows}r");
    write_all_raw(fd, seq.as_bytes());
}

/// Reset the scroll region to the full screen *without* relocating the
/// cursor.
///
/// `DECSTBM` (`CSI r`) always moves the cursor to the home position
/// (top-left), so a bare `ESC [ r` yanks the host cursor to the top
/// after the child exits — the exact symptom `--terminal-passthrough`
/// could never fix, because the reset is unconditional. Wrapping it in a
/// `DECSC`/`DECRC` save-restore makes it cursor-neutral. The save slot is
/// safe to clobber: this only runs once the child has exited, so no
/// in-flight TUI owns it (issue #40 follow-up).
fn reset_scroll_region(fd: i32) {
    write_all_raw(fd, b"\x1b[s\x1b[r\x1b[u");
}

pub fn parse_resize_redraw_key(spec: &str) -> Result<Option<Vec<u8>>, String> {
    let normalized = spec
        .trim()
        .to_ascii_lowercase()
        .replace(['_', '+', ' '], "-");
    if normalized.is_empty()
        || matches!(normalized.as_str(), "off" | "none" | "disabled")
    {
        return Ok(None);
    }

    let parts: Vec<&str> =
        normalized.split('-').filter(|p| !p.is_empty()).collect();
    if parts.len() < 2 {
        return Err("expected values like ctrl-l or disabled".into());
    }

    let mut has_ctrl = false;
    let mut has_shift = false;
    let mut key = None;
    for part in parts {
        match part {
            "ctrl" | "control" => has_ctrl = true,
            "shift" => has_shift = true,
            k if k.len() == 1 && k.as_bytes()[0].is_ascii_alphabetic() => {
                key = Some(k.as_bytes()[0].to_ascii_uppercase());
            }
            other => {
                return Err(format!("unsupported modifier or key {other:?}"));
            }
        }
    }

    if !has_ctrl {
        return Err("only ctrl-based redraw keys are supported".into());
    }
    let Some(key) = key else {
        return Err("missing final letter key".into());
    };
    if has_shift {
        // Shift is accepted for config ergonomics, but terminals
        // collapse Ctrl+Shift+<letter> to the same control byte as
        // Ctrl+<letter>, so both spellings send the same sequence.
    }
    Ok(Some(vec![key & 0x1f]))
}

/// Mutable state shared across one run of the PTY IO loop.
///
/// Bundled into one struct so the per-section helpers can take a
/// single `&mut IoLoop<'_>` instead of half a dozen mutable refs.
struct IoLoop<'a> {
    parser: vt100::Parser,
    prev_screen: vt100::Screen,
    content_rows: u16,
    content_cols: u16,
    pending_redraw: bool,
    pending_resize_redraw_at: Option<Instant>,
    was_alt_screen: bool,
    /// Raw stdout fd. Owned by the process; we just use it for writes.
    stdout: i32,
    /// PTY master fd. Lifetime tied to the parent `pty::run` scope.
    master_raw: i32,
    /// Initial terminal size. Used as fallback when `real_term_size()`
    /// fails inside a SIGWINCH handler.
    init_rows: u16,
    init_cols: u16,
    /// Optional control sequence to inject after the child has had a
    /// chance to process SIGWINCH. None when disabled.
    resize_redraw_key: Option<&'a [u8]>,
    /// Extracts OSC sequences (window title, clipboard, color queries)
    /// from the child stream so they survive the alt-screen vt100
    /// re-render (which drops them).
    osc: OscForwarder,
    terminal_passthrough: bool,
    primary_filter: TerminalFilter,
    /// Filtered-passthrough alt screen (`TerminalFilter::raw_alt`): keeps
    /// all CSI (the host answers DA/kitty/6 n) but drops OSC/DCS strings.
    /// Used on the default (no `--terminal-passthrough`) raw alt path so
    /// fullscreen TUIs render natively without leaking clipboard OSC 52.
    alt_filter: TerminalFilter,
    status_bar: bool,
    /// `CSI 6 n` split across reads of the child PTY.
    cursor_requests: CursorRequestParser,
}

impl<'a> IoLoop<'a> {
    fn new(
        master_raw: i32,
        init_rows: u16,
        init_cols: u16,
        resize_redraw_key: Option<&'a [u8]>,
        terminal_passthrough: bool,
        status_bar: bool,
    ) -> Self {
        // Eagerly touch the trace so a user who set `AI_JAIL_TRACE_SIGWINCH`
        // gets an immediate `trace.log` (proving the running binary actually
        // has tracing compiled in) instead of an empty/missing file when no
        // resize happens to fire. No-op when the env var is unset.
        trace_sigwinch(&format!(
            "ai-jail: io loop started (rows={} cols={} status_bar={} passthrough={})",
            init_rows, init_cols, status_bar, terminal_passthrough
        ));
        let content_rows = if status_bar { init_rows - 1 } else { init_rows };
        let content_cols = init_cols;
        let parser = vt100::Parser::new(content_rows, content_cols, 0);
        let prev_screen = parser.screen().clone();
        Self {
            parser,
            prev_screen,
            content_rows,
            content_cols,
            pending_redraw: false,
            pending_resize_redraw_at: None,
            was_alt_screen: false,
            stdout: nix::libc::STDOUT_FILENO,
            master_raw,
            init_rows,
            init_cols,
            resize_redraw_key,
            osc: OscForwarder::new(terminal_passthrough),
            terminal_passthrough,
            primary_filter: TerminalFilter::new(),
            alt_filter: TerminalFilter::raw_alt(),
            status_bar,
            cursor_requests: CursorRequestParser::default(),
        }
    }

    /// Push existing terminal content into scrollback, clear the
    /// visible area, and set the scroll region so the status bar
    /// row stays untouched by child output.
    fn prime_terminal(&self) {
        if !self.status_bar {
            return;
        }
        let pos = format!("\x1b[{};1H", self.init_rows);
        write_all_raw(self.stdout, pos.as_bytes());
        for _ in 0..self.content_rows {
            write_all_raw(self.stdout, b"\n");
        }
        write_all_raw(self.stdout, b"\x1b[H\x1b[J");
        set_scroll_region(self.stdout, self.content_rows);
    }

    /// Drain a pending SIGWINCH: resize vt100 first, repaint the
    /// real terminal, then resize the PTY and forward SIGWINCH to
    /// the child. Doing it in this order avoids a ghost status-bar
    /// row when the user maximises and avoids a blank flash on the
    /// primary screen.
    fn handle_sigwinch(&mut self) {
        if !SIGWINCH_PENDING.swap(false, Ordering::SeqCst) {
            return;
        }
        let (rows, cols) =
            real_term_size().unwrap_or((self.init_rows, self.init_cols));
        if rows == 0 || (self.status_bar && rows < 2) {
            return;
        }
        let old_content_rows = self.content_rows;
        self.content_rows = if self.status_bar { rows - 1 } else { rows };
        self.content_cols = cols;
        self.parser
            .screen_mut()
            .set_size(self.content_rows, self.content_cols);

        let screen = self.parser.screen();
        let on_alt = screen.alternate_screen();
        if on_alt && self.status_bar {
            // Only the status-bar alt screen is re-rendered by the proxy: it
            // must clear and repaint (the bar overlays the bottom row). Any
            // no-status-bar alt screen — whether raw via
            // `--terminal-passthrough` or the default filtered-raw path — is
            // owned by the child and redraws itself after SIGWINCH, so
            // repainting from vt100 here would race it and reintroduce the
            // cursor drift / "messed up until clear" residue seen on
            // fullscreen TUIs like pool.
            write_all_raw(self.stdout, b"\x1b[r\x1b[H\x1b[J");
            let output = screen.state_formatted();
            write_all_raw(self.stdout, &output);
        } else {
            // Primary screen (e.g. `bash`/`echo`): preserve visible content
            // and let the host terminal re-flow its scrollback into any
            // newly-grown rows the same way a normal terminal would, then let
            // the child repaint on its own after SIGWINCH.
            //
            // Do NOT emit a bare `DECSTBM` reset (`ESC [ r`) on a GROW here:
            // `DECSTBM` relocates the cursor to home and resets the scroll
            // region, which disrupts the host's scrollback re-flow and
            // reintroduces the "residual text" ghost. Emitting a clear of the
            // grown rows (`ESC [ <old+1>;1H ESC [ J`) instead "fixes" the ghost
            // by wiping those rows -- but that also erases the scrolled command
            // output (e.g. `ls`/`seq` that exceeded the old height) that the
            // host would otherwise re-flow, producing "the output disappears".
            // Leaving the host untouched on grow reproduces plain `bash` exactly.
            let grew = old_content_rows < self.content_rows;
            trace_sigwinch(&format!(
                "sigwinch: alt={} status_bar={} old={} new={} grew={}",
                on_alt, self.status_bar, old_content_rows, self.content_rows, grew
            ));
            if self.status_bar {
                // Status bar owns the bottom row: reset the region, blank the
                // old ghost row, and re-establish a status-bar region.
                write_all_raw(self.stdout, b"\x1b[r");
                if grew {
                    let seq = format!("\x1b[{};1H\x1b[2K", old_content_rows + 1);
                    write_all_raw(self.stdout, seq.as_bytes());
                }
                set_scroll_region(self.stdout, self.content_rows);
            } else if !grew {
                // Shimmed or shrunk (or unchanged): a bare `DECSTBM` reset is
                // safe here -- no new rows to re-flow, so it can't disrupt the
                // host. Keep the cursor-neutral reset used on exit.
                reset_scroll_region(self.stdout);
            }
            if !grew {
                // On shrink/same: nudge the host cursor to the parser's
                // cursor so subsequent pass-through output stays in sync.
                let (row, col) = screen.cursor_position();
                let seq = format!("\x1b[{};{}H", row + 1, col + 1);
                write_all_raw(self.stdout, seq.as_bytes());
                trace_sigwinch(&format!(
                    "host_emit: primary CUP {} {}",
                    row + 1,
                    col + 1
                ));
            } else {
                // On grow: do NOT reposition the host cursor. The terminal
                // emulator re-flows its scrollback on resize, so bash's
                // repaint lands at the new bottom -- exactly like plain bash
                // after a `TIOCSWINSZ`+`SIGWINCH`. A CUP to the *stale*
                // parser cursor row (the position under the OLD size) would
                // strand the cursor mid-screen, which is the zoom cursor bug:
                // the cursor sits where the prompt *used* to be instead of at
                // the new bottom. Leaving the host untouched on grow
                // reproduces plain `bash` exactly.
                trace_sigwinch("host_emit: primary GROW (no CUP; host re-flows)");
            }
        }

        self.prev_screen = screen.clone();
        self.was_alt_screen = on_alt;
        if self.status_bar {
            crate::statusbar::update_terminal_state(screen);
            crate::statusbar::redraw();
        }
        resize_pty();
        forward_sigwinch();
        // Arm the post-resize child-byte capture (env-gated) so the trace
        // can show whether the child repainted after the resize/SIGWINCH.
        if sigwinch_trace_enabled() {
            RESIZE_CHILD_BUDGET.store(8192, Ordering::SeqCst);
        }
        if !on_alt && self.resize_redraw_key.is_some() {
            // Let the child process SIGWINCH first, then nudge it to
            // repaint once the terminal goes quiet.
            self.pending_resize_redraw_at =
                Some(Instant::now() + RESIZE_REDRAW_DELAY);
        }
        if self.status_bar {
            let _ = crate::statusbar::take_requests();
        }
        self.pending_redraw = false;
    }

    /// One read from the PTY master: hybrid rendering depending on
    /// alt-screen state and screen transitions. Returns
    /// `ControlFlow::Break(())` when the loop should exit (child
    /// closed the PTY).
    fn handle_master_read(&mut self, buf: &[u8]) -> std::ops::ControlFlow<()> {
        let rem = RESIZE_CHILD_BUDGET.load(Ordering::Relaxed);
        if rem > 0 {
            let n = buf.len().min(rem);
            trace_sigwinch(&format!(
                "child_read after resize: {} bytes: {:?}",
                n,
                &buf[..n]
            ));
            RESIZE_CHILD_BUDGET.fetch_sub(n, Ordering::Relaxed);
        }
        let replies = cursor_replies(
            &mut self.parser,
            &mut self.cursor_requests,
            buf,
            true,
        );
        let screen = self.parser.screen();
        let now_alt = screen.alternate_screen();
        // A "raw" alt screen is one the child owns outright AND there is no
        // status bar overlaying it: every fullscreen TUI (pool, opencode, …)
        // on a terminal WITHOUT a status bar expects the *real* terminal to
        // render it — whether or not `--terminal-passthrough` is set.
        // Re-rendering such output through vt100's state_diff drifts the
        // cursor (the line-number gutter bleeds into the content and columns
        // walk) once the screen wraps or the child redraws after a resize —
        // so on a raw alt screen we forward the child's bytes to the host and
        // let foot/tmux paint them, exactly as a terminal running the TUI
        // directly would.
        //
        // With `--terminal-passthrough` the child's bytes go verbatim;
        // without it we still run them through the `alt_filter` so clipboard
        // OSC 52 stays stripped (the sandboxed child must not read/write the
        // host clipboard) while every CSI is kept (the host answers DA/kitty/
        // `6 n`), and `osc.feed` re-injects the window title sanitized.
        let raw_alt = now_alt && !self.status_bar;
        let was_raw_alt = self.was_alt_screen && !self.status_bar;
        // Answer from the vt100 cursor unless the host will answer
        // itself (a raw alt screen, or the primary screen with
        // passthrough on).
        if !host_answers_cursor_request(
            self.terminal_passthrough,
            self.was_alt_screen,
            now_alt,
            self.status_bar,
        ) && !replies.is_empty()
        {
            write_all_raw(self.master_raw, &replies);
        }

        if raw_alt || was_raw_alt {
            // Raw alt screen: the child owns the terminal. Forward its
            // bytes to the host so foot/tmux renders the TUI directly —
            // do NOT re-render via vt100 (state_diff drifts after a resize)
            // and do NOT intercept the child's CSI capability queries
            // (the host answers DA/kitty/`6 n`, so the TUI detects the real
            // terminal instead of a vt100 cursor that can desync from it).
            //
            // With `--terminal-passthrough` the bytes go verbatim. Without
            // it we filter through `alt_filter` first: that keeps every CSI
            // (drawing + queries, forwarded to the host) but drops OSC/DCS
            // *strings* — so clipboard OSC 52 never reaches the host (the
            // sandboxed child cannot read/write it) while `osc.feed`
            // re-injects only a sanitized window title. `cursor_replies`
            // above still advances the parser so alternate_screen() and the
            // resize path stay correct.
            if self.terminal_passthrough {
                write_all_raw(self.stdout, buf);
            } else {
                let filtered = self.alt_filter.feed(buf);
                write_all_raw(self.stdout, &filtered);
                self.osc.feed(self.stdout, buf);
            }
            self.was_alt_screen = now_alt;
            self.prev_screen = screen.clone();
            return std::ops::ControlFlow::Continue(());
        }

        // Forward CSI control/query sequences that vt100 silently
        // drops (kitty keyboard protocol, Device Attributes, XTVERSION
        // capability queries). Only needed on alt screen paths — the
        // primary screen uses raw pass-through anyway.
        if (now_alt || now_alt != self.was_alt_screen)
            && self.terminal_passthrough
        {
            forward_terminal_queries(self.stdout, buf);
        }

        // Forward OSC sequences the same way. vt100 parses the OSC
        // string but never re-emits it via state_diff / state_formatted,
        // so on the alt screen a fullscreen TUI loses its window-title
        // updates (OSC 0/1/2 — the tab keeps showing the launch command
        // instead of "opencode", #57), clipboard writes (OSC 52), and
        // color queries (OSC 10/11/12). The primary screen's raw
        // pass-through carries these natively; without passthrough the
        // primary filter strips every OSC, so the forwarder runs there
        // too and puts back the sanitized window title.
        if now_alt || !self.terminal_passthrough {
            self.osc.feed(self.stdout, buf);
        }

        if now_alt != self.was_alt_screen {
            // Alt screen transition: full re-render.
            if now_alt {
                write_all_raw(self.stdout, b"\x1b[r");
            } else {
                if self.status_bar {
                    set_scroll_region(self.stdout, self.content_rows);
                }
            }
            write_all_raw(self.stdout, b"\x1b[H\x1b[J");
            let output = screen.state_formatted();
            write_all_raw(self.stdout, &output);
            self.was_alt_screen = now_alt;
        } else if now_alt {
            // Alt screen: cursor-addressed diff.
            let diff = screen.state_diff(&self.prev_screen);
            write_all_raw(self.stdout, &diff);
        } else {
            // Primary screen: raw pass-through for natural scrollback.
            if self.terminal_passthrough {
                write_all_raw(self.stdout, buf);
            } else {
                let filtered = self.primary_filter.feed(buf);
                write_all_raw(self.stdout, &filtered);
            }
            // Re-establish scroll region in case child output contained
            // a reset. Only inject when the output ends at ground state
            // — otherwise our escapes corrupt an in-progress CSI/OSC.
            // DECSTBM (\x1b[1;Nr) moves the cursor home as a side
            // effect; restore via absolute CUP, NOT DECSC/DECRC (which
            // on macOS also save/restore scroll margins and would undo
            // the repair).
            if self.status_bar && ends_at_ground_state(buf) {
                let (row, col) = self.parser.screen().cursor_position();
                set_scroll_region(self.stdout, self.content_rows);
                let seq = format!("\x1b[{};{}H", row + 1, col + 1);
                write_all_raw(self.stdout, seq.as_bytes());
            }
        }

        self.prev_screen = screen.clone();
        if self.status_bar {
            crate::statusbar::update_terminal_state(screen);
            self.pending_redraw = true;
        }
        if self.resize_redraw_key.is_some()
            && self.pending_resize_redraw_at.is_some()
        {
            // Child is still talking — push the redraw-key injection
            // out so we don't fire while the screen is actively
            // changing.
            self.pending_resize_redraw_at =
                Some(Instant::now() + RESIZE_REDRAW_DELAY);
        }
        std::ops::ControlFlow::Continue(())
    }

    /// Drain any remaining bytes on POLLHUP/POLLERR, render the final
    /// state, and reset the scroll region. The loop exits after this.
    fn drain_on_hup(&mut self, buf: &mut [u8]) {
        loop {
            match nix::unistd::read(self.master_raw, buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    self.parser.process(&buf[..n]);
                }
            }
        }
        let screen = self.parser.screen();
        let diff = screen.state_diff(&self.prev_screen);
        write_all_raw(self.stdout, &diff);
        // Reset the scroll region (undo the status-bar region, or any
        // custom region the child left behind) without snapping the
        // cursor home — see [`reset_scroll_region`].
        reset_scroll_region(self.stdout);
        if self.status_bar {
            crate::statusbar::redraw();
        }
    }

    /// Status bar / pending redraw-key flush, run when the child is
    /// quiet (no POLLIN on master).
    fn flush_when_idle(&mut self) {
        if self.status_bar && self.pending_redraw {
            crate::statusbar::redraw();
            self.pending_redraw = false;
        }
        self.maybe_inject_resize_key();
    }

    /// Fire the pending resize-redraw key if its deadline has passed.
    fn maybe_inject_resize_key(&mut self) {
        if let Some(deadline) = self.pending_resize_redraw_at
            && Instant::now() >= deadline
        {
            if let Some(key) = self.resize_redraw_key {
                write_all_raw(self.master_raw, key);
            }
            self.pending_resize_redraw_at = None;
        }
    }
}

fn io_loop(
    master: &OwnedFd,
    init_rows: u16,
    init_cols: u16,
    resize_redraw_key: Option<&[u8]>,
    terminal_passthrough: bool,
    status_bar: bool,
) {
    let stdin_fd = std::io::stdin().as_raw_fd();
    let master_raw = master.as_raw_fd();
    let stdin_bfd = unsafe { BorrowedFd::borrow_raw(stdin_fd) };
    let master_bfd = unsafe { BorrowedFd::borrow_raw(master_raw) };
    let mut buf = [0u8; 8192];

    let mut state = IoLoop::new(
        master_raw,
        init_rows,
        init_cols,
        resize_redraw_key,
        terminal_passthrough,
        status_bar,
    );
    if status_bar {
        crate::statusbar::update_terminal_state(state.parser.screen());
    }
    state.prime_terminal();

    loop {
        state.handle_sigwinch();

        let mut fds = [
            PollFd::new(stdin_bfd, PollFlags::POLLIN),
            PollFd::new(master_bfd, PollFlags::POLLIN),
        ];
        if status_bar && crate::statusbar::take_requests() {
            state.pending_redraw = true;
        }

        match poll(&mut fds, PollTimeout::from(100_u16)) {
            Ok(0) => {
                state.flush_when_idle();
                continue;
            }
            Err(nix::errno::Errno::EINTR) => continue,
            Err(_) => break,
            Ok(_) => {}
        }

        // Check master (child output) first for responsiveness.
        let mut should_break = false;
        if let Some(revents) = fds[1].revents() {
            if revents.contains(PollFlags::POLLIN) {
                match nix::unistd::read(master_raw, &mut buf) {
                    Ok(0) => {
                        should_break = true;
                    }
                    Ok(n) => {
                        let _ = state.handle_master_read(&buf[..n]);
                    }
                    Err(nix::errno::Errno::EINTR) => {}
                    Err(nix::errno::Errno::EIO) => should_break = true,
                    Err(_) => should_break = true,
                }
            }
            if !should_break
                && (revents.contains(PollFlags::POLLHUP)
                    || revents.contains(PollFlags::POLLERR))
            {
                state.drain_on_hup(&mut buf);
                should_break = true;
            }
        }
        if should_break {
            break;
        }

        // Status bar / resize-key flush when child is quiet.
        let child_quiet = !matches!(
            fds[1].revents(),
            Some(r) if r.contains(PollFlags::POLLIN)
        );
        if child_quiet {
            state.flush_when_idle();
        }

        // stdin (user input) → PTY master.
        if let Some(revents) = fds[0].revents()
            && revents.contains(PollFlags::POLLIN)
        {
            match nix::unistd::read(stdin_fd, &mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if terminal_passthrough
                        || !looks_like_terminal_reply(&buf[..n])
                    {
                        write_all_raw(master_raw, &buf[..n]);
                    }
                }
                Err(nix::errno::Errno::EINTR) => {}
                Err(_) => break,
            }
        }
    }

    // Defensive terminal-mode reset (issue #40). The raw-mode guard
    // restores termios separately on Drop.
    crate::output::terminal_reset();
}

/// Decide whether a CSI sequence with the given final byte and
/// optional private-prefix byte (`>`, `<`, `?`) is a terminal query
/// or keyboard-protocol control that vt100 swallows and which must be
/// forwarded verbatim to the real terminal.
///
/// Forwarded:
///   - final `u` with a `>`/`<`/`?` prefix — kitty keyboard protocol
///     (push/pop/query). Without it modifier keys are lost.
///   - final `c` (any prefix) — Primary/Secondary Device Attributes
///     query (`CSI c`, `CSI 0 c`, `CSI > c`). TUIs that depend on the
///     DA reply to detect terminal capabilities otherwise hang waiting
///     and fall back to an ASCII/degraded mode — which drops accented
///     characters (é, ã, …) and box-drawing glyphs (opencode, #57).
///   - final `q` with a `>` prefix — XTVERSION (`CSI > q`) terminal
///     name/version query, used for the same capability detection.
fn csi_final_forwardable(final_b: u8, prefix: Option<u8>) -> bool {
    match final_b {
        b'u' => matches!(prefix, Some(b'>' | b'<' | b'?')),
        b'c' => true,
        b'q' => prefix == Some(b'>'),
        _ => false,
    }
}

/// Parser state for a `CSI 6 n` cursor-position request in child output.
///
/// Drawing commands stay inside vt100. This only notices the request so
/// the proxy can answer it. Forwarding it to the real terminal would
/// race the cursor the proxy itself draws (status bar, scroll region).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum CursorReqState {
    #[default]
    Ground,
    Esc,
    /// Saw `ESC [` or an 8-bit CSI, waiting for the first parameter byte.
    Csi,
    /// Saw a parameter that is exactly `6`, with no other bytes yet.
    Param6,
    /// Some other CSI. Wait for its final byte.
    Skip,
    /// OSC, DCS, APC, or PM. A `CSI 6 n` in a string is not a query.
    String,
    /// Saw `ESC` inside a string, waiting for ST (`\`) or another byte.
    StringEsc,
}

/// Incomplete `CSI 6 n` carried across reads of the child PTY.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CursorRequestParser {
    state: CursorReqState,
}

impl CursorRequestParser {
    /// End offsets (exclusive) of each completed `CSI 6 n` in `data`.
    fn observe(&mut self, data: &[u8]) -> Vec<usize> {
        let mut ends = Vec::new();
        for (i, &b) in data.iter().enumerate() {
            self.step(b, i, &mut ends);
        }
        ends
    }

    fn step(&mut self, b: u8, index: usize, ends: &mut Vec<usize>) {
        self.state = match self.state {
            CursorReqState::Ground => self.ground(b),
            CursorReqState::Esc => self.after_esc(b),
            CursorReqState::Csi => self.csi(b),
            CursorReqState::Param6 => self.param6(b, index, ends),
            CursorReqState::Skip => self.skip(b),
            CursorReqState::String => self.string(b),
            CursorReqState::StringEsc => self.string_esc(b),
        };
    }

    fn ground(&self, b: u8) -> CursorReqState {
        match b {
            0x1b => CursorReqState::Esc,
            0x9b => CursorReqState::Csi,
            0x90 | 0x9d | 0x9e | 0x9f => CursorReqState::String,
            _ => CursorReqState::Ground,
        }
    }

    fn after_esc(&self, b: u8) -> CursorReqState {
        match b {
            b'[' => CursorReqState::Csi,
            b']' | b'P' | b'X' | b'^' | b'_' => CursorReqState::String,
            0x1b => CursorReqState::Esc,
            _ => CursorReqState::Ground,
        }
    }

    fn csi(&self, b: u8) -> CursorReqState {
        match b {
            b'6' => CursorReqState::Param6,
            0x1b => CursorReqState::Esc,
            0x40..=0x7e => CursorReqState::Ground,
            _ => CursorReqState::Skip,
        }
    }

    fn param6(
        &self,
        b: u8,
        index: usize,
        ends: &mut Vec<usize>,
    ) -> CursorReqState {
        match b {
            b'n' => {
                ends.push(index + 1);
                CursorReqState::Ground
            }
            0x1b => CursorReqState::Esc,
            0x40..=0x7e => CursorReqState::Ground,
            _ => CursorReqState::Skip,
        }
    }

    fn skip(&self, b: u8) -> CursorReqState {
        match b {
            0x1b => CursorReqState::Esc,
            0x40..=0x7e => CursorReqState::Ground,
            _ => CursorReqState::Skip,
        }
    }

    fn string(&self, b: u8) -> CursorReqState {
        match b {
            0x07 | 0x9c => CursorReqState::Ground,
            0x1b => CursorReqState::StringEsc,
            _ => CursorReqState::String,
        }
    }

    fn string_esc(&self, b: u8) -> CursorReqState {
        match b {
            b'\\' | 0x9c => CursorReqState::Ground,
            0x1b => CursorReqState::StringEsc,
            _ => CursorReqState::String,
        }
    }
}

/// CPR for a 0-based vt100 cursor. The terminal report is 1-based.
fn cursor_position_report(row: u16, col: u16) -> Vec<u8> {
    format!("\x1b[{};{}R", row + 1, col + 1).into_bytes()
}

/// Feed `data` to `parser` and build one CPR per completed `CSI 6 n`.
///
/// The cursor is read when the request completes, so later bytes in the
/// same chunk do not change that report. `synthesize` is false when the
/// real terminal will answer (primary screen with passthrough): the
/// screen and the scanner still advance, and no local report is built.
fn cursor_replies(
    parser: &mut vt100::Parser,
    requests: &mut CursorRequestParser,
    data: &[u8],
    synthesize: bool,
) -> Vec<u8> {
    let ends = requests.observe(data);
    if !synthesize || ends.is_empty() {
        parser.process(data);
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0;
    for end in ends {
        parser.process(&data[start..end]);
        start = end;
        let (row, col) = parser.screen().cursor_position();
        out.extend(cursor_position_report(row, col));
    }
    if start < data.len() {
        parser.process(&data[start..]);
    }
    out
}

/// The real terminal answers `CSI 6 n` itself whenever the child's
/// bytes reach it: on any no-status-bar alt screen (the child owns the
/// terminal) — entering, steady, or exiting — plus the primary screen
/// with `--terminal-passthrough`. On every other path the proxy owns the
/// cursor and answers from vt100: the primary screen without passthrough
/// (the primary filter strips the query), and any status-bar alt screen
/// (re-rendered through `state_diff`).
///
/// Note this is *not* gated on `terminal_passthrough` for the alt clause:
/// the default (no flag) alt path is now a filtered passthrough that
/// forwards `CSI 6 n` to the host, so the host must answer it the same way
/// it would under `--terminal-passthrough`. Gating it on passthrough would
/// make the proxy synthesise a second (racing) report from a vt100 cursor
/// the host never sees.
fn host_answers_cursor_request(
    passthrough: bool,
    was_alt: bool,
    now_alt: bool,
    status_bar: bool,
) -> bool {
    // Any no-status-bar alt screen (default filtered-raw or passthrough):
    // the child's `CSI 6 n` reaches the host, so the host answers. This
    // covers raw alt enter / steady / exit. The `!was_alt && !now_alt`
    // clause covers the primary screen with passthrough (raw primary).
    (!status_bar && (now_alt || was_alt))
        || (passthrough && !was_alt && !now_alt)
}

/// Scan child output for CSI control/query sequences that vt100 does
/// not re-emit through `state_diff()` / `state_formatted()` and forward
/// them verbatim to the real terminal. Covers the kitty keyboard
/// protocol and terminal capability queries (Device Attributes,
/// XTVERSION). See [`csi_final_forwardable`] for the exact set.
///
/// Only needed on the alt screen — the primary screen uses raw
/// pass-through, which carries these natively.
fn forward_terminal_queries(fd: i32, data: &[u8]) {
    // Tiny state machine:  0=ground  1=ESC  2=CSI-start  3=params
    let mut st: u8 = 0;
    let mut start: usize = 0;
    let mut prefix: Option<u8> = None;
    for (i, &b) in data.iter().enumerate() {
        match st {
            0 => {
                if b == 0x1b {
                    start = i;
                    prefix = None;
                    st = 1;
                }
            }
            1 => {
                if b == b'[' {
                    st = 2;
                } else {
                    st = 0;
                }
            }
            2 => {
                // First byte after CSI — check for a private prefix.
                if b == b'>' || b == b'<' || b == b'?' {
                    prefix = Some(b);
                    st = 3;
                } else if (0x40..=0x7e).contains(&b) {
                    if csi_final_forwardable(b, prefix) {
                        write_all_raw(fd, &data[start..=i]);
                    }
                    st = 0;
                } else {
                    st = 3; // params
                }
            }
            3 => {
                if (0x40..=0x7e).contains(&b) {
                    // Final byte
                    if csi_final_forwardable(b, prefix) {
                        write_all_raw(fd, &data[start..=i]);
                    }
                    st = 0;
                }
                // else: still in params
            }
            _ => st = 0,
        }
    }
}

/// Conservative recognition of terminal replies read from the real terminal.
/// These are only useful after a query was deliberately forwarded.
fn looks_like_terminal_reply(data: &[u8]) -> bool {
    (data.starts_with(b"\x1b[") || data.starts_with(&[0x9b]))
        && data.last().is_some_and(|b| matches!(b, b'R' | b'c' | b'n'))
        || data.starts_with(b"\x1bP")
        || data
            .first()
            .is_some_and(|b| matches!(b, 0x90 | 0x9d | 0x9e | 0x9f))
}

enum FilterState {
    Ground,
    Esc,
    Csi,
    String,
    StringEsc,
    /// Inside `ESC %` — a character-set designation, dropped whole.
    Charset,
}

/// Streaming primary-screen filter. It retains normal text and CSI display
/// controls while dropping OSC, DCS, APC, PM and terminal capability queries.
///
/// The child's output is UTF-8. A byte in `0x80..=0x9f` is only an 8-bit C1
/// control when it stands on its own; inside a multi-byte character it is a
/// continuation byte and must pass through untouched. The block characters
/// U+2590..U+259F (`▐▛▜▝…`, the Claude Code logo) encode as `E2 96 90..9F`,
/// so without that distinction the filter mistook them for DCS/CSI/OSC
/// introducers and a C1 ST, swallowing rows of the banner and leaking the
/// tail of the window-title OSC as text.
/// How aggressively the filter strips CSI queries from the child stream.
enum StripMode {
    /// Primary screen: keep drawing commands (SGR/cursor) + plain text,
    /// drop everything vt100 owns — OSC/DCS/APC/PM strings and the
    /// capability queries the proxy itself answers (kitty-keyboard
    /// `CSI >…u`, DA `CSI c`, XTVERSION `CSI > q`, DSR `CSI 6 n`).
    Primary,
    /// Raw alt screen (filtered passthrough): drop only the OSC/DCS/APC/PM
    /// *strings* (clipboard OSC 52 stays stripped for sandbox security;
    /// the window title is re-emitted sanitized by `OscForwarder`), but
    /// keep **every** CSI — including capability queries — so the real
    /// terminal (foot/tmux) answers them and the fullscreen TUI detects it.
    /// This is the secure default alt path: the child's drawing reaches
    /// the host verbatim (no vt100 re-render, so no resize drift) while
    /// the clipboard never does.
    RawAlt,
}

struct TerminalFilter {
    state: FilterState,
    pending: Vec<u8>,
    /// Continuation bytes still expected for the UTF-8 character in
    /// progress; while non-zero, `0x80..=0xbf` is character data.
    utf8_remaining: u8,
    mode: StripMode,
}

impl TerminalFilter {
    fn new() -> Self {
        Self {
            state: FilterState::Ground,
            pending: Vec::new(),
            utf8_remaining: 0,
            mode: StripMode::Primary,
        }
    }

    /// Filtered-passthrough alt screen: keep all CSI (the host answers
    /// queries), drop only OSC/DCS/APC/PM strings.
    fn raw_alt() -> Self {
        Self {
            state: FilterState::Ground,
            pending: Vec::new(),
            utf8_remaining: 0,
            mode: StripMode::RawAlt,
        }
    }

    /// Track UTF-8 sequence boundaries. Returns `true` when `b` is part
    /// of a multi-byte character (lead or continuation byte) and must be
    /// treated as plain data rather than a C1 control.
    fn utf8_char_byte(&mut self, b: u8) -> bool {
        if self.utf8_remaining > 0 {
            if (0x80..=0xbf).contains(&b) {
                self.utf8_remaining -= 1;
                return true;
            }
            // Truncated sequence: `b` stands on its own.
            self.utf8_remaining = 0;
        }
        self.utf8_remaining = match b {
            0xc2..=0xdf => 1,
            0xe0..=0xef => 2,
            0xf0..=0xf4 => 3,
            _ => 0,
        };
        self.utf8_remaining > 0
    }

    fn feed(&mut self, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(data.len());
        for &b in data {
            if self.utf8_char_byte(b) {
                // Character data: never an introducer or terminator.
                match self.state {
                    FilterState::Ground => out.push(b),
                    FilterState::Esc => {
                        self.pending.push(b);
                        out.extend_from_slice(&self.pending);
                        self.pending.clear();
                        self.state = FilterState::Ground;
                    }
                    FilterState::Csi => {
                        self.pending.push(b);
                        if self.pending.len() > OSC_MAX_LEN {
                            self.pending.clear();
                            self.state = FilterState::Ground;
                        }
                    }
                    FilterState::String => {}
                    FilterState::StringEsc => {
                        self.state = FilterState::String;
                    }
                    // Payload bytes here are ASCII in practice; drop any
                    // stray byte with the rest of the sequence.
                    FilterState::Charset => self.state = FilterState::Ground,
                }
                continue;
            }
            // A fresh ESC outside a string aborts any partial sequence and
            // starts a new one, and the abandoned bytes are never forwarded.
            // Terminals collapse `ESC ESC` and `ESC [ ESC` into a single
            // introducer, so forwarding the partial (the previous behaviour)
            // let `ESC ESC ]52;…BEL` and `ESC [ ESC ]…` smuggle an OSC past
            // the filter (advisory-class terminal-injection). Inside a string
            // ESC may be part of the ST terminator, so String/StringEsc keep
            // their own handling below.
            if b == 0x1b
                && !matches!(
                    self.state,
                    FilterState::String | FilterState::StringEsc
                )
            {
                self.pending.clear();
                self.pending.push(b);
                self.state = FilterState::Esc;
                continue;
            }
            match self.state {
                FilterState::Ground => {
                    if b == 0x1b {
                        self.pending.push(b);
                        self.state = FilterState::Esc;
                    } else if b == 0x9b {
                        self.pending.push(b);
                        self.state = FilterState::Csi;
                    } else if matches!(b, 0x90 | 0x9d | 0x9e | 0x9f) {
                        self.state = FilterState::String;
                    } else {
                        out.push(b);
                    }
                }
                FilterState::Esc => match b {
                    b'[' => {
                        self.pending.push(b);
                        self.state = FilterState::Csi;
                    }
                    // `ESC % @` returns the terminal to ISO 8859-1, where
                    // 0x80..=0x9f are C1 controls again. Everything below
                    // assumes the terminal stays in UTF-8 mode — that is what
                    // makes it safe to pass a continuation byte through as
                    // character data — so the sequence that revokes the
                    // assumption has to go. Dropped rather than forwarded:
                    // ai-jail always speaks UTF-8 to the terminal.
                    b'%' => {
                        self.pending.clear();
                        self.state = FilterState::Charset;
                    }
                    b']' | b'P' | b'X' | b'^' | b'_' => {
                        self.pending.clear();
                        self.state = FilterState::String;
                    }
                    _ => {
                        self.pending.push(b);
                        out.extend_from_slice(&self.pending);
                        self.pending.clear();
                        self.state = FilterState::Ground;
                    }
                },
                // `ESC % / F` designates a multi-byte set; `/` means one
                // more byte follows. Anything else ends the sequence.
                FilterState::Charset => {
                    if b != b'/' {
                        self.state = FilterState::Ground;
                    }
                }
                FilterState::Csi => {
                    self.pending.push(b);
                    if (0x40..=0x7e).contains(&b) {
                        // RawAlt forwards every CSI verbatim — SGR, cursor
                        // moves, DA, kitty-keyboard protocol, DSR `6 n` —
                        // so the real terminal renders the child's drawing
                        // and answers the queries a vt100 re-render cannot
                        // (a vt100 cursor desyncs from the host after a
                        // resize). Primary still drops the queries it owns.
                        if matches!(self.mode, StripMode::RawAlt)
                            || (!csi_final_forwardable(b, {
                                self.pending
                                    .get(2)
                                    .copied()
                                    .filter(|p| matches!(p, b'>' | b'<' | b'?'))
                            }) && b != b'n')
                        {
                            out.extend_from_slice(&self.pending);
                        }
                        self.pending.clear();
                        self.state = FilterState::Ground;
                    } else if self.pending.len() > OSC_MAX_LEN {
                        self.pending.clear();
                        self.state = FilterState::Ground;
                    }
                }
                FilterState::String => match b {
                    // 0x07 (BEL) and 0x9c (C1 ST) both terminate OSC.
                    // Without the 0x9c arm a C1-introduced string (0x9d/
                    // 0x90/0x9e/0x9f) closed by a C1 ST would swallow
                    // every subsequent byte — a child-output DoS and a
                    // filter-bypass surface.
                    0x07 | 0x9c => self.state = FilterState::Ground,
                    0x1b => self.state = FilterState::StringEsc,
                    _ => {}
                },
                FilterState::StringEsc => {
                    self.state = if b == b'\\' || b == 0x9c {
                        // ESC \ (7-bit ST) or a raw C1 ST.
                        FilterState::Ground
                    } else if b == 0x1b {
                        FilterState::StringEsc
                    } else {
                        FilterState::String
                    };
                }
            }
        }
        out
    }
}

/// Hard cap on a single captured OSC sequence. Clipboard payloads
/// (OSC 52) are base64 (~4/3 the raw size) and routinely span several
/// read buffers; this guards against unbounded growth on malformed
/// input.
const OSC_MAX_LEN: usize = 1 << 20;

/// OSC command numbers forwarded unconditionally (set or query) when
/// vt100 swallows them on the alt screen:
///   - 0/1/2 — set icon name / window title. Without forwarding, a
///     fullscreen TUI's tab title never updates (opencode, #57).
///   - 52    — clipboard manipulation (copy / query).
///
/// Color commands (OSC 10/11/12) are handled separately in
/// [`OscForwarder::is_forwardable`] — only their *query* form is
/// forwarded.
const OSC_FORWARD_CMDS: &[&[u8]] = &[b"0", b"1", b"2", b"52"];

/// Longest window title (in bytes) forwarded without
/// `--terminal-passthrough`; longer ones are dropped.
const OSC_TITLE_MAX_LEN: usize = 1024;

/// Where the OSC scanner is within a (possibly read-spanning) sequence.
enum OscState {
    /// Not inside a candidate sequence.
    Ground,
    /// Saw `ESC`, waiting for the `]` that starts an OSC.
    Esc,
    /// Inside the OSC body, accumulating until the terminator.
    Body,
    /// Saw `ESC` in the body — maybe the start of an ST (`ESC \`).
    BodyEsc,
}

/// Streaming extractor for OSC sequences that vt100 drops.
///
/// vt100 consumes the OSC string but does not expose or re-emit it, so
/// on the alt-screen rendering paths (`state_diff` / `state_formatted`)
/// a TUI's window-title update, clipboard write, or color query is
/// silently lost. This scans the child's output byte-by-byte and
/// forwards each complete `ESC ] <cmd> ; … (BEL | ST)` verbatim to the
/// real terminal when `<cmd>` is forwardable (see
/// [`OscForwarder::is_forwardable`]).
///
/// State persists across reads on purpose: a base64 OSC 52 selection
/// can easily exceed a single 8 KiB read, so a stateless per-read scan
/// (like [`forward_terminal_queries`]) would miss split sequences.
///
/// Without `--terminal-passthrough` only window titles get through, and
/// only as rebuilt by [`OscForwarder::sanitized_title`]: clipboard
/// access and color queries stay behind the opt-in.
struct OscForwarder {
    state: OscState,
    buf: Vec<u8>,
    passthrough: bool,
}

impl OscForwarder {
    fn new(passthrough: bool) -> Self {
        Self {
            state: OscState::Ground,
            buf: Vec::new(),
            passthrough,
        }
    }

    fn reset(&mut self) {
        self.state = OscState::Ground;
        self.buf.clear();
    }

    /// Forward the captured sequence only if its OSC command number is
    /// in the allowlist (with passthrough) or it is a safe window title
    /// (without); always reset afterwards.
    fn flush(&mut self, fd: i32) {
        if self.passthrough {
            if Self::is_forwardable(&self.buf) {
                write_all_raw(fd, &self.buf);
            }
        } else if let Some(title) = Self::sanitized_title(&self.buf) {
            write_all_raw(fd, &title);
        }
        self.reset();
    }

    /// A complete OSC 0/1/2 (set icon name / window title) rebuilt as
    /// `ESC ] <cmd> ; <text> BEL`, or `None` unless the text is valid
    /// UTF-8 of at most [`OSC_TITLE_MAX_LEN`] bytes with no C0, DEL or
    /// C1 control. Setting a title is display-only; reading one back
    /// (`CSI 21 t`) is a query and stays filtered.
    fn sanitized_title(buf: &[u8]) -> Option<Vec<u8>> {
        let body = buf.strip_prefix(b"\x1b]")?;
        let body = body
            .strip_suffix(b"\x07")
            .or_else(|| body.strip_suffix(b"\x1b\\"))?;
        let sep = body.iter().position(|&b| b == b';')?;
        let (cmd, text) = (&body[..sep], &body[sep + 1..]);
        if !matches!(cmd, b"0" | b"1" | b"2") || text.len() > OSC_TITLE_MAX_LEN
        {
            return None;
        }
        let text = std::str::from_utf8(text).ok()?;
        if text.chars().any(char::is_control) {
            return None;
        }
        let mut out = Vec::with_capacity(body.len() + 3);
        out.extend_from_slice(b"\x1b]");
        out.extend_from_slice(cmd);
        out.push(b';');
        out.extend_from_slice(text.as_bytes());
        out.push(0x07);
        Some(out)
    }

    /// Inspect a captured `ESC ] <cmd> (;|terminator) …` buffer and
    /// decide whether `<cmd>` is one we forward.
    fn is_forwardable(buf: &[u8]) -> bool {
        // Strip the leading "\x1b]" introducer.
        let body = match buf.strip_prefix(b"\x1b]") {
            Some(b) => b,
            None => return false,
        };
        // The command number runs up to the first ';' (or the
        // terminator, for parameter-less commands).
        let end = body
            .iter()
            .position(|&b| b == b';' || b == 0x07 || b == 0x1b)
            .unwrap_or(body.len());
        let cmd = &body[..end];
        if OSC_FORWARD_CMDS.contains(&cmd) {
            return true;
        }
        // Color commands (fg/bg/cursor). Forward only the *query* form
        // (`\x1b]1{0,1,2};?…`): forwarding a set would change the real
        // terminal's default color, and the matching reset (OSC
        // 110/111/112) is not forwarded, so a set would leak past the
        // child's exit. The query reply round-trips via our stdin.
        if matches!(cmd, b"10" | b"11" | b"12") {
            return body.get(end) == Some(&b';')
                && body.get(end + 1) == Some(&b'?');
        }
        false
    }

    fn feed(&mut self, fd: i32, data: &[u8]) {
        for &b in data {
            self.push(fd, b);
        }
    }

    fn push(&mut self, fd: i32, b: u8) {
        match self.state {
            OscState::Ground => {
                if b == 0x1b {
                    self.buf.push(b);
                    self.state = OscState::Esc;
                }
            }
            OscState::Esc => {
                if b == b']' {
                    self.buf.push(b);
                    self.state = OscState::Body;
                } else if b == 0x1b {
                    // Another ESC — keep waiting for the ']'.
                    self.buf.clear();
                    self.buf.push(b);
                } else {
                    self.reset();
                }
            }
            OscState::Body => match b {
                0x07 => {
                    // BEL terminator.
                    self.buf.push(b);
                    self.flush(fd);
                }
                0x1b => {
                    self.buf.push(b);
                    self.state = OscState::BodyEsc;
                }
                _ => {
                    self.buf.push(b);
                    if self.buf.len() > OSC_MAX_LEN {
                        self.reset();
                    }
                }
            },
            OscState::BodyEsc => {
                if b == b'\\' {
                    // ST terminator (ESC \).
                    self.buf.push(b);
                    self.flush(fd);
                } else if b == 0x1b {
                    // Another ESC — keep waiting for the backslash.
                    self.buf.push(b);
                } else {
                    // Stray ESC, not a terminator: malformed, drop.
                    self.reset();
                }
            }
        }
    }
}

/// Check whether raw output ends at the ground state of the VT
/// escape parser.  If `false`, the buffer ends mid-sequence and
/// injecting our own escape codes would corrupt the child's
/// incomplete CSI/OSC/DCS.
fn ends_at_ground_state(data: &[u8]) -> bool {
    // 0 = ground, 1 = ESC, 2 = CSI params, 3 = string (OSC/DCS)
    let mut st: u8 = 0;
    for &b in data {
        st = match (st, b) {
            (0, 0x1b) => 1,
            (0, _) => 0,
            (1, b'[') => 2,
            (1, b']' | b'P' | b'X' | b'^' | b'_') => 3,
            (1, 0x20..=0x2f) => 1, // intermediates
            (1, _) => 0,           // single-char escape done
            (2, 0x40..=0x7e) => 0, // CSI final byte
            (2, _) => 2,           // params / intermediates
            (3, 0x07) => 0,        // BEL terminates OSC
            (3, 0x1b) => 1,        // possible ST (ESC \)
            (3, _) => 3,
            _ => 0,
        };
    }
    st == 0
}

/// Write all bytes to a raw fd using libc::write (works in all
/// contexts including pre_exec).
fn write_all_raw(fd: i32, data: &[u8]) {
    let mut off = 0;
    while off < data.len() {
        let n = unsafe {
            nix::libc::write(
                fd,
                data[off..].as_ptr().cast::<nix::libc::c_void>(),
                data.len() - off,
            )
        };
        if n <= 0 {
            break;
        }
        off += n as usize;
    }
}

/// Whether stdout is attached to a real terminal, i.e. child output
/// must be filtered for hostile terminal control sequences before it
/// reaches the host terminal. Pipes, files, and other non-terminal
/// sinks do not interpret escape sequences, so filtering can be
/// skipped on those paths.
// Not yet referenced from main.rs (wired in a parallel lane); drop
// this allow once a production caller exists.
#[allow(dead_code)]
pub fn stdout_needs_filter() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
}

/// Config-aware PTY entry point for the explicit terminal passthrough opt-in.
/// A PTY pair allocated before the sandbox command is built.
///
/// The macOS profile scopes its terminal `file-ioctl` grant to this exact
/// device, so the slave path has to be known while the SBPL profile is still
/// being generated — hence allocating here rather than inside [`run`].
pub struct Pty(nix::pty::OpenptyResult);

/// Allocate the PTY pair the sandboxed child will use as its terminal.
pub fn open() -> Result<Pty, String> {
    nix::pty::openpty(None, None)
        .map(Pty)
        .map_err(|e| format!("openpty: {e}"))
}

impl Pty {
    /// Device path of the slave side (for example `/dev/ttys003`).
    ///
    /// `None` when it cannot be resolved; callers then grant no terminal
    /// ioctl access at all rather than widening the rule to every terminal.
    /// Duplicate of the slave fd, for handing to a child as stdio while
    /// this `Pty` stays alive.
    #[cfg(all(test, target_os = "macos"))]
    pub fn slave_try_clone(&self) -> std::io::Result<OwnedFd> {
        self.0.slave.try_clone()
    }

    pub fn slave_path(&self) -> Option<std::path::PathBuf> {
        use std::os::unix::ffi::OsStrExt;
        // ptsname returns a pointer into static storage, copied immediately.
        let raw = unsafe { nix::libc::ptsname(self.0.master.as_raw_fd()) };
        if raw.is_null() {
            return None;
        }
        let bytes = unsafe { std::ffi::CStr::from_ptr(raw) }.to_bytes();
        Some(std::path::PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
    }
}

pub fn run_with_config(
    pty: Pty,
    cmd: &mut std::process::Command,
    resize_redraw_key: Option<&[u8]>,
    config: &crate::config::Config,
    status_bar: bool,
) -> Result<i32, String> {
    run_with_passthrough(
        pty,
        cmd,
        resize_redraw_key,
        config.terminal_passthrough_enabled(),
        status_bar,
    )
}

fn run_with_passthrough(
    pty: Pty,
    cmd: &mut std::process::Command,
    resize_redraw_key: Option<&[u8]>,
    terminal_passthrough: bool,
    status_bar: bool,
) -> Result<i32, String> {
    use std::os::unix::process::CommandExt;

    let (rows, cols) = real_term_size().unwrap_or((24, 80));
    if rows == 0 || (status_bar && rows < 2) {
        return Err("Terminal too small for status bar".into());
    }

    // PTY pair was allocated by the caller (see [`open`]).
    let master = pty.0.master;
    let slave = pty.0.slave;

    // Set FD_CLOEXEC on master so child doesn't inherit it
    let master_raw = master.as_raw_fd();
    unsafe {
        let flags = nix::libc::fcntl(master_raw, nix::libc::F_GETFD);
        nix::libc::fcntl(
            master_raw,
            nix::libc::F_SETFD,
            flags | nix::libc::FD_CLOEXEC,
        );
    }

    // Reserve a row only when the status overlay is active.
    set_initial_size(&master, if status_bar { rows - 1 } else { rows }, cols);

    // Enter raw mode on real stdin
    let saved = enter_raw_mode()?;
    let raw_mode_guard = RawModeGuard::new(saved);

    // Configure child to use PTY slave as stdin/stdout/stderr
    let slave_raw = slave.as_raw_fd();
    // Record the slave PTY device so the SIGWINCH /proc fallback (which
    // matches descendants' `tty_nr` against the device the child actually
    // holds) compares against the slave, not the master (`/dev/ptmx`).
    set_slave_dev(slave_raw);
    unsafe {
        cmd.pre_exec(move || {
            if nix::libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            // `ioctl`'s request argument is `c_ulong` on glibc and
            // macOS but `c_int` on musl; `as _` lets the cast follow
            // whichever the target libc declares.
            if nix::libc::ioctl(slave_raw, nix::libc::TIOCSCTTY as _, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            nix::libc::dup2(slave_raw, 0);
            nix::libc::dup2(slave_raw, 1);
            nix::libc::dup2(slave_raw, 2);
            if slave_raw > 2 {
                nix::libc::close(slave_raw);
            }
            Ok(())
        });
    }

    // Spawn child
    let child = cmd
        .spawn()
        .map_err(|e| format!("Failed to start sandbox: {e}"))?;

    let pid = child.id() as i32;
    crate::signals::set_child_pid(pid);
    MASTER_FD.store(master_raw, Ordering::SeqCst);
    STATUS_BAR_ACTIVE.store(status_bar, Ordering::SeqCst);

    // Close slave in parent — child has its own copy
    drop(slave);

    // Run IO loop (blocks until child exits / master HUP)
    io_loop(
        &master,
        rows,
        cols,
        resize_redraw_key,
        terminal_passthrough,
        status_bar,
    );

    // Clean up
    MASTER_FD.store(-1, Ordering::SeqCst);
    STATUS_BAR_ACTIVE.store(false, Ordering::SeqCst);
    drop(master);
    drop(raw_mode_guard);

    // Wait for child
    let exit_code = crate::signals::wait_child(pid);

    // Prevent double-wait
    std::mem::forget(child);

    Ok(exit_code)
}

fn real_term_size() -> Option<(u16, u16)> {
    let mut ws = unsafe { std::mem::zeroed::<nix::libc::winsize>() };
    let ret = unsafe {
        nix::libc::ioctl(
            nix::libc::STDOUT_FILENO,
            nix::libc::TIOCGWINSZ,
            &mut ws,
        )
    };
    if ret == 0 && ws.ws_row > 0 && ws.ws_col > 0 {
        Some((ws.ws_row, ws.ws_col))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn vt100_screen_tracks_content() {
        let mut parser = vt100::Parser::new(24, 80, 0);
        parser.process(b"Hello, world!");
        let screen = parser.screen();
        let row = screen.rows(0, 80).next().unwrap();
        assert!(row.starts_with("Hello, world!"));
    }

    #[test]
    fn vt100_diff_produces_output() {
        let mut parser = vt100::Parser::new(24, 80, 0);
        let prev = parser.screen().clone();
        parser.process(b"test output");
        let diff = parser.screen().contents_diff(&prev);
        assert!(!diff.is_empty());
    }

    #[test]
    fn vt100_resize_preserves_content() {
        let mut parser = vt100::Parser::new(24, 80, 0);
        parser.process(b"line1\r\nline2\r\nline3");
        parser.screen_mut().set_size(30, 100);
        let screen = parser.screen();
        let row0 = screen.rows(0, 100).next().unwrap();
        assert!(row0.starts_with("line1"));
    }

    #[test]
    fn vt100_primary_screen_repaint_after_resize_contains_content() {
        let mut parser = vt100::Parser::new(4, 20, 0);
        parser.process(b"line1\r\nline2\r\nline3");
        parser.screen_mut().set_size(6, 30);

        let output = parser.screen().state_formatted();

        assert!(!parser.screen().alternate_screen());
        assert!(!output.is_empty());
        assert!(String::from_utf8_lossy(&output).contains("line1"));
        assert!(String::from_utf8_lossy(&output).contains("line2"));
    }

    #[test]
    fn parse_resize_redraw_key_ctrl_l() {
        assert_eq!(
            super::parse_resize_redraw_key("ctrl-l").unwrap(),
            Some(vec![0x0c])
        );
    }

    #[test]
    fn parse_resize_redraw_key_ctrl_shift_l_aliases_ctrl_l() {
        assert_eq!(
            super::parse_resize_redraw_key("ctrl-shift-l").unwrap(),
            Some(vec![0x0c])
        );
    }

    #[test]
    fn parse_resize_redraw_key_disabled() {
        assert_eq!(super::parse_resize_redraw_key("disabled").unwrap(), None);
    }

    #[test]
    fn parse_resize_redraw_key_rejects_unknown() {
        assert!(super::parse_resize_redraw_key("alt-l").is_err());
    }

    #[test]
    fn vt100_state_diff_includes_mode_changes() {
        let mut parser = vt100::Parser::new(24, 80, 0);
        let prev = parser.screen().clone();
        // Enable bracketed paste mode
        parser.process(b"\x1b[?2004h");
        let diff = parser.screen().state_diff(&prev);
        // state_diff includes both content and mode changes
        assert!(parser.screen().bracketed_paste());
        assert!(!prev.bracketed_paste());
        // Diff should contain the mode change sequence
        assert!(!diff.is_empty());
    }

    #[test]
    fn vt100_alt_screen_tracking() {
        let mut parser = vt100::Parser::new(24, 80, 0);
        assert!(!parser.screen().alternate_screen());
        parser.process(b"\x1b[?1049h");
        assert!(parser.screen().alternate_screen());
        parser.process(b"\x1b[?1049l");
        assert!(!parser.screen().alternate_screen());
    }

    #[test]
    fn vt100_mouse_mode_tracking() {
        let mut parser = vt100::Parser::new(24, 80, 0);
        assert_eq!(
            parser.screen().mouse_protocol_mode(),
            vt100::MouseProtocolMode::None
        );
        parser.process(b"\x1b[?1003h");
        assert_eq!(
            parser.screen().mouse_protocol_mode(),
            vt100::MouseProtocolMode::AnyMotion
        );
    }

    #[test]
    fn vt100_cursor_visibility() {
        let mut parser = vt100::Parser::new(24, 80, 0);
        assert!(!parser.screen().hide_cursor());
        parser.process(b"\x1b[?25l");
        assert!(parser.screen().hide_cursor());
        parser.process(b"\x1b[?25h");
        assert!(!parser.screen().hide_cursor());
    }

    use super::ends_at_ground_state;
    use super::forward_terminal_queries;
    use std::os::unix::io::AsRawFd;

    fn capture_kbd_forward(data: &[u8]) -> Vec<u8> {
        let (r, w) = nix::unistd::pipe().unwrap();
        forward_terminal_queries(w.as_raw_fd(), data);
        drop(w);
        let mut out = vec![0u8; 256];
        let n = nix::unistd::read(r.as_raw_fd(), &mut out).unwrap_or(0);
        out.truncate(n);
        out
    }

    fn capture_kbd_forward_enabled(data: &[u8], enabled: bool) -> Vec<u8> {
        let (r, w) = nix::unistd::pipe().unwrap();
        if enabled {
            forward_terminal_queries(w.as_raw_fd(), data);
        }
        drop(w);
        let mut out = vec![0u8; 256];
        let n = nix::unistd::read(r.as_raw_fd(), &mut out).unwrap_or(0);
        out.truncate(n);
        out
    }

    #[test]
    fn kbd_push_mode_forwarded() {
        // CSI > 1 u  — push keyboard mode flags=1
        let out = capture_kbd_forward(b"\x1b[>1u");
        assert_eq!(out, b"\x1b[>1u");
    }

    #[test]
    fn terminal_queries_are_denied_without_passthrough() {
        assert!(capture_kbd_forward_enabled(b"\x1b[>1u", false).is_empty());
    }

    #[test]
    fn kbd_pop_mode_forwarded() {
        // CSI < u  — pop keyboard mode
        let out = capture_kbd_forward(b"\x1b[<u");
        assert_eq!(out, b"\x1b[<u");
    }

    #[test]
    fn kbd_query_mode_forwarded() {
        // CSI ? u  — query keyboard mode
        let out = capture_kbd_forward(b"\x1b[?u");
        assert_eq!(out, b"\x1b[?u");
    }

    #[test]
    fn kbd_mixed_with_other_csi() {
        // SGR + keyboard push + cursor move
        let data = b"\x1b[31m\x1b[>1u\x1b[H";
        let out = capture_kbd_forward(data);
        assert_eq!(out, b"\x1b[>1u");
    }

    #[test]
    fn kbd_no_false_positives() {
        // Regular CSI sequences should not be forwarded
        let out = capture_kbd_forward(b"\x1b[31m\x1b[H\x1b[J");
        assert!(out.is_empty());
    }

    #[test]
    fn kbd_plain_text_ignored() {
        let out = capture_kbd_forward(b"hello world");
        assert!(out.is_empty());
    }

    #[test]
    fn primary_device_attributes_forwarded() {
        // CSI c  — Primary DA query. opencode and other TUIs rely on
        // the reply for capability detection; without forwarding they
        // degrade to ASCII and drop accented chars (#57).
        assert_eq!(capture_kbd_forward(b"\x1b[c"), b"\x1b[c");
        // CSI 0 c — same query with an explicit parameter.
        assert_eq!(capture_kbd_forward(b"\x1b[0c"), b"\x1b[0c");
    }

    #[test]
    fn secondary_device_attributes_forwarded() {
        // CSI > c  — Secondary DA query.
        assert_eq!(capture_kbd_forward(b"\x1b[>c"), b"\x1b[>c");
    }

    #[test]
    fn xtversion_query_forwarded() {
        // CSI > q  — XTVERSION terminal name/version query.
        assert_eq!(capture_kbd_forward(b"\x1b[>q"), b"\x1b[>q");
    }

    #[test]
    fn unrelated_csi_finals_not_forwarded() {
        // SGR (m), cursor home (H), erase (J), DSR (n) stay swallowed.
        // vt100 reconstructs drawing commands. Forwarding CSI 6 n would
        // race the cursor the proxy draws on the real terminal. The
        // child is answered from the vt100 cursor in `cursor_replies`.
        assert!(capture_kbd_forward(b"\x1b[31m").is_empty());
        assert!(capture_kbd_forward(b"\x1b[6n").is_empty());
        assert!(capture_kbd_forward(b"\x1b[2J").is_empty());
    }

    #[test]
    fn plain_csi_u_not_forwarded() {
        // `\x1b[u` (no >/</? prefix) is SCORC — restore cursor position,
        // NOT a kitty keyboard control. vt100 handles it on the screen
        // model, so forwarding it would move the real cursor wrongly.
        // This guards the prefix requirement in csi_final_forwardable.
        assert!(capture_kbd_forward(b"\x1b[u").is_empty());
        assert!(capture_kbd_forward(b"\x1b[1;5u").is_empty());
    }

    #[test]
    fn osc_title_split_across_reads_forwarded_whole() {
        // A window-title update can straddle a read boundary just like
        // OSC 52; the forwarder must reassemble and forward it whole.
        let seq = b"\x1b]2;opencode session\x07";
        assert_eq!(capture_osc52(&[&seq[..6], &seq[6..]]), seq);
    }

    fn capture_osc52(chunks: &[&[u8]]) -> Vec<u8> {
        let (r, w) = nix::unistd::pipe().unwrap();
        let mut fwd = super::OscForwarder::new(true);
        for chunk in chunks {
            fwd.feed(w.as_raw_fd(), chunk);
        }
        drop(w);
        let mut out = vec![0u8; 8192];
        let n = nix::unistd::read(r.as_raw_fd(), &mut out).unwrap_or(0);
        out.truncate(n);
        out
    }

    #[test]
    fn osc52_bel_terminated_forwarded() {
        let seq = b"\x1b]52;c;aGVsbG8=\x07";
        assert_eq!(capture_osc52(&[seq]), seq);
    }

    #[test]
    fn osc52_st_terminated_forwarded() {
        let seq = b"\x1b]52;c;aGVsbG8=\x1b\\";
        assert_eq!(capture_osc52(&[seq]), seq);
    }

    #[test]
    fn osc52_query_forwarded() {
        // Apps query the clipboard with `?`; the terminal replies on
        // our stdin, which the IO loop already routes to the child.
        let seq = b"\x1b]52;c;?\x07";
        assert_eq!(capture_osc52(&[seq]), seq);
    }

    #[test]
    fn osc52_split_across_reads_forwarded_whole() {
        // A base64 payload can straddle the 8 KiB read boundary.
        let seq = b"\x1b]52;c;aGVsbG8gd29ybGQ=\x07";
        assert_eq!(capture_osc52(&[&seq[..7], &seq[7..]]), seq);
    }

    #[test]
    fn osc52_embedded_in_other_output_extracted() {
        let data = b"some text\x1b[31m\x1b]52;c;Zm9v\x07more";
        assert_eq!(capture_osc52(&[data]), b"\x1b]52;c;Zm9v\x07");
    }

    #[test]
    fn osc_window_title_forwarded() {
        // OSC 0 (icon + title) and OSC 2 (title) must reach the real
        // terminal so the tab shows the TUI name, not the launch
        // command (#57). vt100 swallows them on the alt screen.
        let t0 = b"\x1b]0;opencode\x07";
        assert_eq!(capture_osc52(&[t0]), t0);
        let t2 = b"\x1b]2;opencode\x1b\\";
        assert_eq!(capture_osc52(&[t2]), t2);
    }

    #[test]
    fn osc_color_query_forwarded() {
        // OSC 11 (background color) query — TUIs adapt their palette
        // from the reply, which round-trips via our stdin.
        let q = b"\x1b]11;?\x07";
        assert_eq!(capture_osc52(&[q]), q);
        // OSC 10 (foreground) and OSC 12 (cursor) queries too.
        let q10 = b"\x1b]10;?\x1b\\";
        assert_eq!(capture_osc52(&[q10]), q10);
    }

    #[test]
    fn osc_color_set_not_forwarded() {
        // A color *set* (no `?`) must NOT be forwarded: it would change
        // the real terminal's default color, and the matching reset
        // (OSC 110/111/112) is not forwarded, so the change would leak
        // past the child's exit.
        let set = b"\x1b]11;rgb:0000/0000/0000\x07";
        assert!(capture_osc52(&[set]).is_empty());
    }

    #[test]
    fn osc_unlisted_command_not_forwarded() {
        // OSC 4 (palette set) is not in the allowlist; vt100 handles
        // palette state itself, so we must not double-forward it.
        let out = capture_osc52(&[b"\x1b]4;1;rgb:ff/00/00\x07"]);
        assert!(out.is_empty());
    }

    #[test]
    fn osc52_plain_text_ignored() {
        assert!(capture_osc52(&[b"just plain text"]).is_empty());
    }

    #[test]
    fn osc52_oversized_payload_dropped() {
        let mut seq = b"\x1b]52;c;".to_vec();
        seq.extend(std::iter::repeat_n(b'A', super::OSC_MAX_LEN + 16));
        seq.push(0x07);
        assert!(capture_osc52(&[&seq]).is_empty());
    }

    fn capture_filtered_osc(chunks: &[&[u8]]) -> Vec<u8> {
        let (r, w) = nix::unistd::pipe().unwrap();
        let mut fwd = super::OscForwarder::new(false);
        for chunk in chunks {
            fwd.feed(w.as_raw_fd(), chunk);
        }
        drop(w);
        let mut out = vec![0u8; 8192];
        let n = nix::unistd::read(r.as_raw_fd(), &mut out).unwrap_or(0);
        out.truncate(n);
        out
    }

    #[test]
    fn osc_clipboard_denied_and_title_kept_without_passthrough() {
        let out = capture_filtered_osc(&[
            b"\x1b]2;title\x07\x1b]52;c;data\x07\x1b]52;c;?\x07",
        ]);
        assert_eq!(out, b"\x1b]2;title\x07");
    }

    #[test]
    fn osc_color_query_denied_without_passthrough() {
        assert!(capture_filtered_osc(&[b"\x1b]11;?\x07"]).is_empty());
    }

    #[test]
    fn osc_title_without_passthrough_is_rebuilt_with_bel() {
        assert_eq!(capture_filtered_osc(&[b"\x1b]0;a\x1b\\"]), b"\x1b]0;a\x07");
        assert_eq!(
            capture_filtered_osc(&[b"\x1b]1;", b"icon\x07"]),
            b"\x1b]1;icon\x07"
        );
        let utf8 = "\x1b]0;\u{2733} Claude Code\x07";
        assert_eq!(capture_filtered_osc(&[utf8.as_bytes()]), utf8.as_bytes());
    }

    #[test]
    fn osc_title_without_passthrough_drops_unsafe_text() {
        // C0 control, DEL, a C1 control encoded as UTF-8, invalid UTF-8.
        for seq in [
            &b"\x1b]0;a\x08b\x07"[..],
            b"\x1b]0;a\x7fb\x07",
            b"\x1b]0;a\xc2\x9bb\x07",
            b"\x1b]0;a\xffb\x07",
        ] {
            assert!(capture_filtered_osc(&[seq]).is_empty(), "{seq:?}");
        }
        let mut long = b"\x1b]2;".to_vec();
        long.extend(std::iter::repeat_n(b'x', super::OSC_TITLE_MAX_LEN + 1));
        long.push(0x07);
        assert!(capture_filtered_osc(&[&long]).is_empty());
    }

    /// Run one child-output chunk through the IO loop and return what
    /// reaches the real terminal.
    fn rendered_output(passthrough: bool, chunk: &[u8]) -> Vec<u8> {
        let (master_r, master_w) = nix::unistd::pipe().unwrap();
        let (out_r, out_w) = nix::unistd::pipe().unwrap();
        let mut io = super::IoLoop::new(
            master_w.as_raw_fd(),
            24,
            80,
            None,
            passthrough,
            false,
        );
        io.stdout = out_w.as_raw_fd();
        let _ = io.handle_master_read(chunk);
        drop(out_w);
        drop((master_r, master_w));
        let mut out = vec![0u8; 8192];
        let n = nix::unistd::read(out_r.as_raw_fd(), &mut out).unwrap_or(0);
        out.truncate(n);
        out
    }

    #[test]
    fn primary_screen_forwards_title_but_not_clipboard_by_default() {
        let out = rendered_output(
            false,
            b"before\x1b]0;my title\x07\x1b]52;c;ZGF0YQ==\x07after",
        );
        let text = String::from_utf8_lossy(&out);
        assert!(text.contains("\x1b]0;my title\x07"), "{text:?}");
        assert!(!text.contains("]52;"), "{text:?}");
        assert!(
            text.contains("before") && text.contains("after"),
            "{text:?}"
        );
    }

    #[test]
    fn alt_screen_forwards_title_but_not_clipboard_by_default() {
        let out = rendered_output(
            false,
            b"\x1b[?1049h\x1b]2;tui\x07\x1b]52;c;ZGF0YQ==\x07",
        );
        let text = String::from_utf8_lossy(&out);
        assert!(text.contains("\x1b]2;tui\x07"), "{text:?}");
        assert!(!text.contains("]52;"), "{text:?}");
    }

    #[test]
    fn primary_screen_title_passes_through_once_with_passthrough() {
        let out = rendered_output(true, b"\x1b]0;raw\x07");
        assert_eq!(out, b"\x1b]0;raw\x07");
    }

    #[test]
    fn primary_filter_keeps_display_controls_and_strips_queries() {
        let mut filter = super::TerminalFilter::new();
        let out =
            filter.feed(b"text\x1b[31mred\x1b[2;4H\x1b]52;c;data\x07\x1b[c");
        assert_eq!(out, b"text\x1b[31mred\x1b[2;4H");
    }

    #[test]
    fn primary_filter_strips_c1_terminal_strings_and_queries() {
        let mut filter = super::TerminalFilter::new();
        let out = filter.feed(b"text\x9d52;c;data\x07\x9b?1;2cvisible");
        assert_eq!(out, b"textvisible");
        assert!(super::looks_like_terminal_reply(b"\x9b12;4R"));
        assert!(super::looks_like_terminal_reply(b"\x9d52;c;reply\x07"));
    }

    #[test]
    fn primary_filter_c1_st_terminates_c1_osc() {
        // C1 OSC (0x9d) closed by C1 ST (0x9c): text after the
        // terminator must survive instead of being swallowed forever.
        let mut filter = super::TerminalFilter::new();
        let out = filter.feed(b"text\x9d2;title\x9cafter");
        assert_eq!(out, b"textafter");
    }

    #[test]
    fn primary_filter_c1_st_terminates_all_c1_introducers() {
        // DCS (0x90), OSC (0x9d), PM (0x9e), APC (0x9f): every C1
        // string introducer must end on a raw C1 ST (0x9c).
        for introducer in [0x90u8, 0x9d, 0x9e, 0x9f] {
            let mut filter = super::TerminalFilter::new();
            let seq = [introducer, b'p', b'!', 0x9c, b'o', b'k'];
            assert_eq!(
                filter.feed(&seq),
                b"ok",
                "introducer {introducer:#04x} not terminated by C1 ST"
            );
        }
    }

    #[test]
    fn primary_filter_c1_st_split_across_reads() {
        // The C1 ST may land in the next read buffer; the filter state
        // must persist and still terminate the string.
        let mut filter = super::TerminalFilter::new();
        assert_eq!(filter.feed(b"\x9d0;evil"), b"");
        assert_eq!(filter.feed(b"title\x9cresumed"), b"resumed");
    }

    #[test]
    fn primary_filter_output_resumes_after_c1_st() {
        // After a C1 ST terminator the filter is back at ground state:
        // plain text and CSI display controls pass through again.
        let mut filter = super::TerminalFilter::new();
        let out = filter.feed(b"\x9fpayload\x9c\x1b[31mred");
        assert_eq!(out, b"\x1b[31mred");
    }

    #[test]
    fn primary_filter_c1_st_after_string_esc() {
        // ESC inside a string followed by a raw C1 ST also terminates
        // (treat 0x9c as ST in all string states).
        let mut filter = super::TerminalFilter::new();
        let out = filter.feed(b"\x9d0;t\x1b\x9cok");
        assert_eq!(out, b"ok");
    }

    #[test]
    fn primary_filter_doubled_esc_does_not_leak_osc() {
        // Advisory-class terminal injection: terminals collapse ESC ESC and
        // ESC [ ESC into one introducer, so a following OSC would reach the
        // terminal. The filter must not forward the OSC introducer.
        for prefix in [&b"\x1b"[..], &b"\x1b["[..]] {
            let mut f = super::TerminalFilter::new();
            let mut input = prefix.to_vec();
            input.extend_from_slice(b"\x1b]52;c;QUJD\x07");
            let out = f.feed(&input);
            assert!(
                !out.windows(3).any(|w| w == b"\x1b]5"),
                "OSC introducer leaked after {prefix:?}: {out:?}"
            );
            // And no raw ] 5 2 payload leaks as text either.
            assert!(
                !out.windows(4).any(|w| w == b"]52;"),
                "OSC payload leaked as text after {prefix:?}: {out:?}"
            );
        }
    }

    #[test]
    fn primary_filter_drops_charset_designation() {
        // Treating a UTF-8 continuation byte as character data is only safe
        // while the terminal is in UTF-8 mode. `ESC % @` returns it to
        // ISO 8859-1, where 0x80..=0x9f are C1 controls again — so an agent
        // could send that, then smuggle a C1 CSI through as the third byte
        // of a UTF-8 sequence and have the terminal act on it. Dropping the
        // designation keeps the assumption the filter relies on true.
        let mut filter = super::TerminalFilter::new();
        assert_eq!(
            filter.feed(b"\x1b%@"),
            b"",
            "ESC % @ must not reach the terminal"
        );
        assert_eq!(
            filter.feed(b"\x1b%G"),
            b"",
            "ESC % G must not reach the terminal"
        );
        // `ESC % / F` designates a multi-byte set: three bytes after ESC.
        assert_eq!(
            filter.feed(b"\x1b%/4"),
            b"",
            "ESC % / F must not reach the terminal"
        );
        // Text on either side is untouched.
        assert_eq!(filter.feed(b"ok\x1b%@done"), b"okdone");
        // And a lone ESC % mid-stream does not swallow what follows.
        assert_eq!(filter.feed(b"\x1b%@\xe2\x96\x9c"), b"\xe2\x96\x9c");
    }

    #[test]
    fn primary_filter_passes_utf8_block_characters_through() {
        // U+2590..U+259F encode as E2 96 90..9F: the continuation bytes
        // coincide with the C1 DCS (0x90), CSI (0x9b), ST (0x9c), OSC
        // (0x9d) codes. The Claude Code logo is drawn with them and
        // used to lose whole rows (issue seen on macOS startup).
        let mut filter = super::TerminalFilter::new();
        let logo = " ▐\x1b[48;2;0;0;0m▛████▜█\x1b[12G\x1b[1mClaude\x1b[19GCode"
            .as_bytes();
        assert_eq!(filter.feed(logo), logo);
        let row2 = "▝▜\x1b[48;2;0;0;0m█████\x1b[49m█▀\x1b[12GFable".as_bytes();
        assert_eq!(filter.feed(row2), row2);
        // Four-byte characters carry 0x9f-style bytes as well.
        assert_eq!(filter.feed("😀 ok".as_bytes()), "😀 ok".as_bytes());
    }

    #[test]
    fn primary_filter_utf8_character_split_across_reads() {
        let mut filter = super::TerminalFilter::new();
        assert_eq!(filter.feed(b"\xe2\x96"), b"\xe2\x96");
        // 0x9c here is the last byte of U+259C, not a C1 ST.
        assert_eq!(filter.feed(b"\x9c rest"), b"\x9c rest");
    }

    #[test]
    fn primary_filter_drops_osc_title_with_utf8_payload_whole() {
        // OSC 0 with "✳" (E2 9C B3) in the title: the 0x9c continuation
        // byte used to terminate the string early and leak
        // "\xb3 Claude Code\x07" to the terminal.
        let mut filter = super::TerminalFilter::new();
        let out = filter.feed("\x1b]0;✳ Claude Code\x07after".as_bytes());
        assert_eq!(out, b"after");
    }

    #[test]
    fn primary_filter_c1_after_complete_utf8_char_still_recognized() {
        // A standalone 0x9d following a finished character is still an
        // 8-bit OSC introducer and gets stripped up to its terminator.
        let mut filter = super::TerminalFilter::new();
        let out = filter.feed(b"\xc3\xa9\x9d0;evil\x07ok");
        assert_eq!(out, b"\xc3\xa9ok");
    }

    #[test]
    fn terminal_reply_detection_is_conservative() {
        assert!(super::looks_like_terminal_reply(b"\x1b[12;4R"));
        assert!(!super::looks_like_terminal_reply(b"\x1b[A"));
    }

    fn cursor_replies(chunks: &[&[u8]], synthesize: bool) -> Vec<u8> {
        let mut parser = vt100::Parser::new(24, 80, 0);
        let mut requests = super::CursorRequestParser::default();
        let mut out = Vec::new();
        for chunk in chunks {
            out.extend(super::cursor_replies(
                &mut parser,
                &mut requests,
                chunk,
                synthesize,
            ));
        }
        out
    }

    #[test]
    fn cursor_request_is_answered_from_vt100_origin() {
        // CSI 6 n at the home position. The report is 1-based.
        assert_eq!(cursor_replies(&[b"\x1b[6n"], true), b"\x1b[1;1R");
    }

    #[test]
    fn cursor_request_uses_position_at_the_request() {
        // CUP is 1-based. Bytes after the request must not move the
        // reported position. "ab" leaves the cursor in column 3.
        assert_eq!(cursor_replies(&[b"\x1b[8;3H\x1b[6n"], true), b"\x1b[8;3R");
        assert_eq!(cursor_replies(&[b"ab\x1b[6ncd"], true), b"\x1b[1;3R");
        assert_eq!(
            cursor_replies(&[b"\x1b[6n\x1b[4;5H\x1b[6n"], true),
            b"\x1b[1;1R\x1b[4;5R"
        );
    }

    #[test]
    fn cursor_request_ignores_other_status_reports() {
        // CSI 5 n is device status, not the cursor. Extra digits,
        // a private prefix, or another parameter are not CSI 6 n.
        assert!(cursor_replies(&[b"\x1b[5n"], true).is_empty());
        assert!(cursor_replies(&[b"\x1b[16n"], true).is_empty());
        assert!(cursor_replies(&[b"\x1b[?6n"], true).is_empty());
        assert!(cursor_replies(&[b"\x1b[6;1n"], true).is_empty());
    }

    #[test]
    fn cursor_request_split_across_reads_is_answered_once() {
        assert_eq!(
            cursor_replies(&[b"\x1b[2;4H\x1b[6", b"n"], true),
            b"\x1b[2;4R"
        );
    }

    #[test]
    fn cursor_request_inside_osc_is_not_a_query() {
        assert!(cursor_replies(&[b"\x1b]0;\x1b[6n\x07"], true).is_empty());
        assert_eq!(
            cursor_replies(&[b"\x1b]0;title\x07\x1b[6n"], true),
            b"\x1b[1;1R"
        );
    }

    #[test]
    fn cursor_request_stays_silent_when_the_host_answers() {
        // Primary screen + passthrough forwards the query raw. A second
        // report from the vt100 model would race that reply. The cursor
        // still moves, so a later local answer uses that position.
        let mut parser = vt100::Parser::new(24, 80, 0);
        let mut requests = super::CursorRequestParser::default();
        let silent = super::cursor_replies(
            &mut parser,
            &mut requests,
            b"\x1b[3;3H\x1b[6n",
            false,
        );
        assert!(silent.is_empty());
        let answered =
            super::cursor_replies(&mut parser, &mut requests, b"\x1b[6n", true);
        assert_eq!(answered, b"\x1b[3;3R");
    }

    #[test]
    fn child_pty_receives_the_cursor_report() {
        let (read, write) = nix::unistd::pipe().unwrap();
        let mut io =
            super::IoLoop::new(write.as_raw_fd(), 24, 80, None, false, false);
        let _ = io.handle_master_read(b"\x1b[4;2H\x1b[6n");
        drop(write);
        let mut buf = [0u8; 32];
        let n = nix::unistd::read(read.as_raw_fd(), &mut buf).unwrap();
        assert_eq!(&buf[..n], b"\x1b[4;2R");
    }

    #[test]
    fn host_answers_cursor_request_semantics() {
        // The real terminal answers `CSI 6 n` whenever the child's bytes
        // reach it: on ANY no-status-bar alt screen (the default
        // filtered-raw path forwards the query through `alt_filter`, and
        // `--terminal-passthrough` forwards it verbatim) — entering,
        // steady, or exiting — plus the primary screen with passthrough.
        // The proxy answers from its vt100 cursor everywhere else:
        // primary without passthrough (the primary filter strips the
        // query) and any status-bar alt screen (re-rendered through
        // state_diff). Gating the alt clause on passthrough would make
        // the proxy synthesise a second (racing) report on the default
        // path, since the host never sees the query it is answering.
        let bar = false;
        // Primary + passthrough: raw -> host answers.
        assert!(super::host_answers_cursor_request(true, false, false, bar));
        // Primary, no passthrough: proxy answers (primary_filter strips 6 n).
        assert!(!super::host_answers_cursor_request(false, false, false, bar));
        // Default raw alt (no passthrough, no status bar): the alt filter
        // keeps `CSI 6 n` and forwards it to the host, so the host answers.
        assert!(super::host_answers_cursor_request(false, false, true, bar)); // entry
        assert!(super::host_answers_cursor_request(false, true, true, bar)); // steady
        assert!(super::host_answers_cursor_request(false, true, false, bar)); // exit
        // Passthrough raw alt: host answers.
        assert!(super::host_answers_cursor_request(true, false, true, bar));
        assert!(super::host_answers_cursor_request(true, true, true, bar));
        assert!(super::host_answers_cursor_request(true, true, false, bar));
        // Status-bar alt (re-rendered): proxy answers, any passthrough.
        assert!(!super::host_answers_cursor_request(true, false, true, true));
        assert!(!super::host_answers_cursor_request(true, true, true, true));
        assert!(!super::host_answers_cursor_request(true, true, false, true));
    }

    #[test]
    fn primary_filter_drops_cursor_position_request() {
        let mut filter = super::TerminalFilter::new();
        assert_eq!(filter.feed(b"\x1b[6nvisible"), b"visible");
    }

    #[test]
    fn ground_state_plain_text() {
        assert!(ends_at_ground_state(b"hello world"));
    }

    #[test]
    fn ground_state_complete_csi() {
        // Complete SGR: \x1b[31m
        assert!(ends_at_ground_state(b"\x1b[31m"));
        // Complete 24-bit color
        assert!(ends_at_ground_state(b"\x1b[38;2;255;0;0mRed"));
    }

    #[test]
    fn ground_state_incomplete_csi() {
        // Ends mid-CSI (no final byte yet)
        assert!(!ends_at_ground_state(b"\x1b[38;2;255"));
        // Just ESC [
        assert!(!ends_at_ground_state(b"\x1b["));
        // Just ESC
        assert!(!ends_at_ground_state(b"\x1b"));
    }

    #[test]
    fn ground_state_text_then_incomplete() {
        assert!(!ends_at_ground_state(b"hello\x1b[31"));
    }

    #[test]
    fn ground_state_osc_complete() {
        // OSC terminated by BEL
        assert!(ends_at_ground_state(b"\x1b]0;title\x07"));
        // OSC terminated by ST (ESC \)
        assert!(ends_at_ground_state(b"\x1b]0;title\x1b\\"));
    }

    #[test]
    fn ground_state_osc_incomplete() {
        assert!(!ends_at_ground_state(b"\x1b]0;title"));
    }

    #[test]
    fn ground_state_empty() {
        assert!(ends_at_ground_state(b""));
    }

    #[test]
    fn ground_state_single_char_escape() {
        // ESC 7 (DECSC) is a complete single-char escape
        assert!(ends_at_ground_state(b"\x1b7"));
    }

    #[test]
    fn ground_state_scroll_region_reset() {
        // \x1b[r resets scroll margins — full CSI, ends at ground
        assert!(ends_at_ground_state(b"\x1b[r"));
        // Table border followed by scroll region reset
        assert!(ends_at_ground_state(
            b"\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\x1b[r"
        ));
    }

    #[test]
    fn allocated_pty_resolves_a_slave_device_path() {
        // macOS scopes its terminal file-ioctl grant to exactly this path.
        // If ptsname ever stopped resolving, the profile would silently
        // emit no terminal ioctl rule and agents could not enter raw mode.
        let pty = super::open().expect("openpty");
        let path = pty.slave_path().expect("slave path");
        assert!(
            path.starts_with("/dev/"),
            "unexpected slave device path: {}",
            path.display()
        );
        assert!(
            path.exists(),
            "slave device does not exist: {}",
            path.display()
        );
    }

    #[test]
    fn ground_state_cup_sequence() {
        // CUP (\x1b[row;colH) used to restore cursor position
        // after set_scroll_region — complete CSI, ends at ground.
        assert!(ends_at_ground_state(b"\x1b[12;1H"));
        assert!(ends_at_ground_state(b"\x1b[1;80H"));
    }

    /// Assert the host-repaint bytes `handle_sigwinch` emits for a
    /// primary-screen (status_bar off — the tmux default) resize GROW.
    ///
    /// On a grow the proxy must **not** touch the grown rows: no `DECSTBM`
    /// reset (`ESC [ r`) and no erase (`ESC [ <old+1>;1H ESC [ J`). A bare
    /// `ESC [ r` relocates the cursor home and resets the scroll region,
    /// which disrupts the host terminal's scrollback re-flow and reintroduces
    /// the "residual text after tmux fullscreen" ghost. Clearing the grown
    /// rows instead "fixes" the ghost by wiping those rows — but that also
    /// erases the scrolled command output (`ls`/`seq` that exceeded the old
    /// height) that a normal terminal re-flows into them, producing
    /// "the output disappears". Leaving the host untouched on grow reproduces
    /// plain `bash` exactly (verified: `seq 1 30` + zoom re-flows 9..30 into
    /// rows 1-22, prompt at row 24).
    ///
    /// Deterministic in `cargo test`: the harness captures fd 1 (a pipe), so
    /// `real_term_size()` is `None` and the fallback `init_rows` is used,
    /// giving a 12 -> 24 grow regardless of the real terminal size.
    #[test]
    fn handle_sigwinch_does_not_blank_grown_rows_on_primary_grow() {
        let (_master_r, master_w) = nix::unistd::pipe().unwrap();
        let (out_r, out_w) = nix::unistd::pipe().unwrap();
        // init_rows=24 is the fallback host size; seed content_rows=12 so the
        // resize is a grow.
        let mut io = super::IoLoop::new(
            master_w.as_raw_fd(),
            24,
            80,
            None,
            false,
            false,
        );
        io.stdout = out_w.as_raw_fd();
        io.content_rows = 12;
        io.content_cols = 80;
        io.parser.screen_mut().set_size(12, 80);
        io.parser.process(b"\x1b[5;3H"); // cursor -> row 5, col 3

        super::set_sigwinch_pending();
        io.handle_sigwinch();

        drop(out_w);
        drop(master_w);
        let mut buf = vec![0u8; 1024];
        let n = nix::unistd::read(out_r.as_raw_fd(), &mut buf).unwrap_or(0);
        let got = String::from_utf8_lossy(&buf[..n]);

        // No erasing of the grown rows (would wipe scrolled command output).
        assert!(
            !got.contains("\x1b[J") && !got.contains("\x1b[2K"),
            "must not clear rows/lines on primary grow; got {got:?}"
        );
        // On a primary GROW we emit NOTHING to the host. The terminal
        // emulator re-flows its scrollback on resize, so bash's repaint after
        // SIGWINCH lands at the new bottom -- exactly like plain `bash`. A CUP
        // to the *stale* parser cursor row (the cursor position under the OLD
        // size) would strand the cursor mid-screen, which is the zoom cursor
        // bug; leaving the host untouched on grow reproduces plain bash.
        // (On a shrink the only host write is the cursor restore; see the
        // shrink test.)
        assert!(
            got.is_empty(),
            "expected no host write on primary grow, got {got:?}"
        );
    }

    #[test]
    fn handle_sigwinch_emits_cup_and_scroll_reset_on_shrink() {
        // Counterpart to the grow test: a no-status-bar SHRINK must still
        // reset the scroll region (safe -- no re-flow to disrupt) and re-sync
        // the host cursor to the parser cursor so bash's repaint lands in the
        // right place. The grow fix must not have touched this path.
        let (_master_r, master_w) = nix::unistd::pipe().unwrap();
        let (out_r, out_w) = nix::unistd::pipe().unwrap();
        // init_rows=24 host -> content_rows=12 (the fallback host size used
        // when stdout is not a tty); seed a grow so the first handle_sigwinch
        // is a shrink (12 -> ... no, force: set content_rows=24, host=12).
        let mut io = super::IoLoop::new(
            master_w.as_raw_fd(),
            12, // host rows (smaller than content_rows -> shrink)
            80,
            None,
            false,
            false,
        );
        io.stdout = out_w.as_raw_fd();
        io.content_rows = 24; // larger than host (12) -> this resize is a shrink
        io.content_cols = 80;
        io.parser.screen_mut().set_size(24, 80);
        io.parser.process(b"\x1b[5;3H"); // cursor -> row 5, col 3

        super::set_sigwinch_pending();
        io.handle_sigwinch();

        drop(out_w);
        drop(master_w);
        let mut buf = vec![0u8; 1024];
        let n = nix::unistd::read(out_r.as_raw_fd(), &mut buf).unwrap_or(0);
        let got = String::from_utf8_lossy(&buf[..n]);

        // Shrink still resets the scroll region (cursor-neutral DECSTBM) and
        // re-syncs the host cursor to the parser position so bash's repaint
        // lands in the right place. The grow fix must not have touched this.
        assert!(
            got.contains("\x1b[r"),
            "shrink must reset the scroll region; got {got:?}"
        );
        // set_size may preserve (5;3) or home (1;1) the cursor; accept either.
        assert!(
            got.contains("\x1b[5;3H") || got.contains("\x1b[1;1H") || got.contains("\r"),
            "shrink must re-sync the host cursor; got {got:?}"
        );
        // No grown-row erase on shrink either.
        assert!(
            !got.contains("\x1b[J") && !got.contains("\x1b[2K"),
            "must not clear rows/lines on shrink; got {got:?}"
        );
    }

    #[test]
    fn handle_sigwinch_skips_host_repaint_on_raw_alt_screen() {
        // `--terminal-passthrough` alt screen (no status bar): the child
        // owns the terminal and redraws itself after SIGWINCH. handle_sigwinch
        // must NOT repaint the host from vt100 (ESC[r ESC[H ESC[J +
        // state_formatted) — that re-render races the child's own redraw and
        // is the source of pool's post-resize column drift ("messed up until
        // clear"). Asserts the host fd stays empty on a grow.
        let (_master_r, master_w) = nix::unistd::pipe().unwrap();
        let (out_r, out_w) = nix::unistd::pipe().unwrap();
        // terminal_passthrough=true, status_bar=false -> raw alt path.
        let mut io = super::IoLoop::new(
            master_w.as_raw_fd(),
            24,
            80,
            None,
            true,
            false,
        );
        io.stdout = out_w.as_raw_fd();
        // Place the parser in the alt screen without driving the full
        // handle_master_read (which would itself write the enter sequence).
        io.parser.process(b"\x1b[?1049h");
        io.was_alt_screen = true;
        // Seed a stale size so handle_sigwinch is a grow (12 -> 24 via the
        // init_rows fallback used when stdout is a pipe, not a tty).
        io.content_rows = 12;
        io.content_cols = 80;
        io.parser.screen_mut().set_size(12, 80);

        super::set_sigwinch_pending();
        io.handle_sigwinch();

        drop(out_w);
        let mut buf = vec![0u8; 1024];
        let n = nix::unistd::read(out_r.as_raw_fd(), &mut buf).unwrap_or(0);
        assert!(
            n == 0,
            "raw alt resize must not repaint the host; got {buf:?}"
        );
    }

    #[test]
    fn raw_alt_screen_forwards_child_bytes_verbatim() {
        // A fullscreen TUI (alt screen) on a terminal WITHOUT a status bar
        // owns its terminal: ai-jail must forward its bytes verbatim to the
        // host (so foot/tmux renders the TUI natively) instead of
        // re-rendering through vt100's state_diff — which drifts the cursor
        // after a resize (line-number gutter bleeds into content, columns
        // walk). Exercises `handle_master_read` directly via the no-bwrap
        // `run_with_config` io_loop construction (pipe-backed fds).
        let (_master_r, master_w) = nix::unistd::pipe().unwrap();
        let (out_r, out_w) = nix::unistd::pipe().unwrap();
        // status_bar = false, terminal_passthrough = true (--terminal-passthrough).
        let mut io = super::IoLoop::new(
            master_w.as_raw_fd(),
            24,
            80,
            None,
            true,
            false,
        );
        io.stdout = out_w.as_raw_fd();

        // Alt-screen enter + a CSI capability query + an OSC title + body,
        // exactly the kind of launch stream pool/opencode emit.
        let buf = b"\x1b[?1049h\x1b[>4;2mHello, TUI\x1b]2;pool\x07world";
        let _ = io.handle_master_read(buf);

        // Alternate screen state must be remembered across reads so the
        // resize path and the exit transition stay consistent.
        assert!(io.was_alt_screen, "alternate screen should be detected");

        drop(out_w);
        let mut got = vec![0u8; 2048];
        let n = nix::unistd::read(out_r.as_raw_fd(), &mut got).unwrap_or(0);
        got.truncate(n);

        // Raw: child bytes forwarded byte-for-byte — no state_formatted
        // repaint, no ESC[H/ESC[J host-clear prefix, no cursor-restore.
        assert_eq!(
            got,
            buf,
            "raw alt screen must forward child bytes verbatim, got {got:?}"
        );
    }

    #[test]
    fn bar_alt_screen_still_re_renders_via_state_diff() {
        // Regression guard: with a status bar the alt screen is re-rendered
        // (the bar overlays the bottom row), it is NOT raw-forwarded.
        let (_master_r, master_w) = nix::unistd::pipe().unwrap();
        let (out_r, out_w) = nix::unistd::pipe().unwrap();
        let mut io = super::IoLoop::new(
            master_w.as_raw_fd(),
            24,
            80,
            None,
            false,
            true, // status_bar ON
        );
        io.stdout = out_w.as_raw_fd();

        let buf = b"\x1b[?1049hX";
        let _ = io.handle_master_read(buf);

        drop(out_w);
        let mut got = vec![0u8; 4096];
        let n = nix::unistd::read(out_r.as_raw_fd(), &mut got).unwrap_or(0);
        got.truncate(n);

        // Not verbatim: the transition repaint (ESC[r ESC[H ESC[J + state_formatted)
        // is emitted, so the raw child bytes are re-rendered, not echoed.
        assert_ne!(
            got,
            buf,
            "status-bar alt screen must not be raw-forwarded"
        );
        assert!(
            got.starts_with(b"\x1b[r\x1b[H\x1b[J"),
            "status-bar alt entry must repaint the host screen, got {got:?}"
        );
    }

    #[test]
    fn handle_sigwinch_skips_host_repaint_on_default_raw_alt_screen() {
        // Default (no --terminal-passthrough) filtered-raw alt (no status
        // bar): the child still owns the terminal and redraws itself after
        // SIGWINCH, so handle_sigwinch must NOT repaint the host from vt100
        // — exactly like the passthrough raw alt. The host fd stays empty
        // on a grow.
        let (_master_r, master_w) = nix::unistd::pipe().unwrap();
        let (out_r, out_w) = nix::unistd::pipe().unwrap();
        // terminal_passthrough=false, status_bar=false -> default raw alt.
        let mut io = super::IoLoop::new(
            master_w.as_raw_fd(),
            24,
            80,
            None,
            false,
            false,
        );
        io.stdout = out_w.as_raw_fd();
        // Place the parser in the alt screen without driving the full
        // handle_master_read (which would itself write the enter sequence).
        io.parser.process(b"\x1b[?1049h");
        io.was_alt_screen = true;
        // Seed a stale size so handle_sigwinch is a grow (12 -> 24 via the
        // init_rows fallback used when stdout is a pipe, not a tty).
        io.content_rows = 12;
        io.content_cols = 80;
        io.parser.screen_mut().set_size(12, 80);

        super::set_sigwinch_pending();
        io.handle_sigwinch();

        drop(out_w);
        let mut buf = vec![0u8; 1024];
        let n = nix::unistd::read(out_r.as_raw_fd(), &mut buf).unwrap_or(0);
        assert!(
            n == 0,
            "default raw alt resize must not repaint the host; got {buf:?}"
        );
    }

    #[test]
    fn default_raw_alt_screen_forwards_csi_queries_strips_clipboard() {
        // Default (no --terminal-passthrough) filtered-raw alt: the child's
        // CSI capability queries (DA, DSR 6 n, kitty keyboard) reach the host
        // so the TUI detects the real terminal, while clipboard OSC 52 is
        // still stripped — the host answers the queries (no proxy synthesis).
        let out = rendered_output(
            false,
            b"\x1b[?1049h\x1b[c\x1b[6n\x1b[>1u\x1b]52;c;ZGF0YQ==\x07",
        );
        let text = String::from_utf8_lossy(&out);
        // CSI queries forwarded verbatim to the host.
        assert!(text.contains("\x1b[c"), "{text:?}");
        assert!(text.contains("\x1b[6n"), "{text:?}");
        assert!(text.contains("\x1b[>1u"), "{text:?}");
        // Clipboard 52 stays stripped.
        assert!(!text.contains("]52;"), "{text:?}");
    }

    #[test]
    fn default_raw_alt_screen_forwards_drawing_no_state_diff() {
        // Default raw alt forwards the child's actual drawing bytes (no
        // vt100 state_diff repaint) — that is what eliminates the resize
        // drift: the host renders exactly what the child emits.
        let out = rendered_output(false, b"\x1b[?1049h\x1b[2J\x1b[Hrow1");
        assert_eq!(
            out,
            b"\x1b[?1049h\x1b[2J\x1b[Hrow1",
            "default raw alt must forward drawing verbatim, got {out:?}"
        );
    }

    #[test]
    fn raw_alt_filter_keeps_csi_strips_osc_and_dcs() {
        let mut f = super::TerminalFilter::raw_alt();
        // CSI drawing + capability queries (DA, DSR 6 n, kitty keyboard)
        // are kept verbatim so the real terminal renders and answers them.
        assert_eq!(
            f.feed(b"\x1b[31m\x1b[6n\x1b[c\x1b[>1u"),
            b"\x1b[31m\x1b[6n\x1b[c\x1b[>1u"
        );
        // OSC (title + clipboard 52) and DCS are dropped — clipboard 52
        // must never reach the host on the default path.
        assert_eq!(
            f.feed(b"\x1b]2;t\x07\x1b]52;c;data\x07\x1bPdata\x1b\\text"),
            b"text"
        );
        // UTF-8 drawing bytes are not mistaken for C1 introducers.
        let mut g = super::TerminalFilter::raw_alt();
        assert_eq!(
            g.feed(&[0xe2, 0x96, 0x90, b'A']),
            &[0xe2, 0x96, 0x90, b'A']
        );
        // OSC 52 split across reads is still dropped.
        let mut h = super::TerminalFilter::raw_alt();
        assert_eq!(h.feed(b"\x1b]52;c;"), b"");
        assert_eq!(h.feed(b"part\x07after"), b"after");
    }

    #[test]
    fn primary_filter_primary_mode_is_unchanged_and_still_strips_queries() {
        // Regression guard: new StripMode must not alter the existing
        // primary-screen filter (drops DA/kitty/6 n/DCS/OSC; keeps drawing).
        let mut f = super::TerminalFilter::new();
        assert_eq!(
            f.feed(b"\x1b[31m\x1b]52;c;data\x07\x1b[c\x1b[6n"),
            b"\x1b[31m"
        );
    }

    #[test]
    fn child_host_namespace_pgrps_resolves_descendant_tty_holder() {
        // Non-bwrap stand-in for the bwrap case: the recorded child pid is a
        // wrapper (`sh`) and the real tty holder (`sleep`, forked by sh) is its
        // descendant -- exactly mirroring bwrap where ai-jail's direct child
        // is bwrap and the shell is bwrap's descendant. Exercises the same
        // /proc traversal + tty_nr/tpgid parsing the bwrap path relies on,
        // without needing bwrap (which is blocked in this sandbox).
        use std::os::unix::process::CommandExt;

        let pty = match super::open() {
            Ok(p) => p,
            Err(_) => return, // PTY unavailable in this environment -> skip
        };
        let master_fd = pty.0.master.as_raw_fd();
        let slave_raw = pty.0.slave.as_raw_fd();
        let stdin_slave = match pty.0.slave.try_clone() {
            Ok(fd) => fd,
            Err(_) => return,
        };
        let stdout_slave = match pty.0.slave.try_clone() {
            Ok(fd) => fd,
            Err(_) => return,
        };
        let stderr_slave = match pty.0.slave.try_clone() {
            Ok(fd) => fd,
            Err(_) => return,
        };

        let mut cmd = std::process::Command::new("sh");
        cmd.args(["-c", "sleep 3"]);
        cmd.stdin(std::process::Stdio::from(stdin_slave));
        cmd.stdout(std::process::Stdio::from(stdout_slave));
        cmd.stderr(std::process::Stdio::from(stderr_slave));
        unsafe {
            cmd.pre_exec(move || {
                // Mirror pty::run_with_passthrough: become a session leader
                // with the slave as controlling terminal so the forked
                // `sleep` inherits it (so sleep's tty_nr == slave device).
                let _ = nix::libc::setsid();
                let _ =
                    nix::libc::ioctl(slave_raw, nix::libc::TIOCSCTTY as _, 0);
                Ok(())
            })
        };

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(_) => return,
        };
        let pid = child.id() as i32;

        super::MASTER_FD.store(
            master_fd,
            std::sync::atomic::Ordering::SeqCst,
        );
        // Record the slave device so the resolver can do a *precise*
        // tty_nr == slave_dev match (not just the loose fallback).
        super::set_slave_dev(pty.0.slave.as_raw_fd());
        crate::signals::set_child_pid(pid);
        // Give `sh` time to fork `sleep` and take the controlling tty.
        std::thread::sleep(std::time::Duration::from_millis(300));

        let pgrps = super::child_host_namespace_pgrps();
        // The descendant `sleep` holds the ctty, so the resolver MUST return a
        // non-empty set of killable host-namespace pgids. If this fails, the
        // bwrap fallback won't deliver SIGWINCH either.
        assert!(
            !pgrps.is_empty(),
            "expected the resolver to find a tty-holding descendant of {pid} \
             (master_dev logged in trace)"
        );

        let _ = unsafe { nix::libc::kill(pid, nix::libc::SIGTERM) };
        let _ = child.wait();
        super::MASTER_FD.store(-1, std::sync::atomic::Ordering::SeqCst);
        super::SLAVE_DEV.store(0, std::sync::atomic::Ordering::SeqCst);
        crate::signals::set_child_pid(0);
    }
}
