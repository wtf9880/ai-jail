#!/usr/bin/env bash
# No-bwrap end-to-end repro for the grow-CUP fix. Compares the cursor position
# after a terminal GROW (12 -> 48 rows) between plain bash and the ai-jail
# pty::run_with_config io_loop (bash -i, no bwrap). With the fix both land the
# prompt on the BOTTOM row (48); before the fix ai-jail stranded it mid-screen.
set -u
cd "$(dirname "$0")"
TB=$(ls -t target/debug/deps/ai_jail-* 2>/dev/null | grep -v '\.d$' | head -1)
echo "test binary: $TB"
rm -f screen_before_*.txt screen_after_*.txt trace.log

probe_prompt_row() {
  # last 1-based line index whose rendered text ends in a bare '$'
  awk 'prev ~ /\$$/ { row=NR } { prev=$0 } END { print row+0 }' "$1"
}

tmux_kill() { tmux kill-session -t "$1" 2>/dev/null || true; }

# ---- 1) plain bash baseline ----
tmux_kill plain
tmux new-session -d -s plain -x 140 -y 12
tmux send-keys -t plain "seq 1 30" Enter
sleep 0.6
tmux capture-pane -p -t plain > screen_before_plain.txt
tmux resize-window -t plain -y 48 -x 140
sleep 0.8
tmux capture-pane -p -t plain > screen_after_plain.txt
echo "[plain] prompt row before=$(probe_prompt_row screen_before_plain.txt) after=$(probe_prompt_row screen_after_plain.txt) (lines=$(wc -l < screen_after_plain.txt))"
tmux send-keys -t plain "exit" Enter; sleep 0.2; tmux_kill plain

# ---- 2) ai-jail io_loop (test binary) IN the pane, trace enabled ----
tmux_kill aj
tmux new-session -d -s aj -x 140 -y 12
tmux send-keys -t aj "AI_JAIL_TRACE_SIGWINCH=$PWD/trace.log '$TB' repro_bash_zoom_via_pty_proxy --ignored --nocapture" Enter
sleep 1.2
tmux send-keys -t aj "seq 1 30" Enter
sleep 0.5
tmux capture-pane -p -t aj > screen_before.txt
echo "ai-jail pane before: $(tmux display-message -p -t aj '#{pane_height}') rows"
tmux resize-window -t aj -y 48 -x 140
sleep 1.0
echo "ai-jail pane after:  $(tmux display-message -p -t aj '#{pane_height}') rows"
tmux capture-pane -p -t aj > screen_after.txt
echo "[ai-jail] prompt row before=$(probe_prompt_row screen_before.txt) after=$(probe_prompt_row screen_after.txt) (lines=$(wc -l < screen_after.txt))"
tmux send-keys -t aj "exit" Enter; sleep 0.4; tmux_kill aj

echo "=== trace.log ==="; grep -E "sigwinch:|host_emit:|forward_sigwinch:|child_read after resize" trace.log 2>/dev/null | head -12
echo "=== ai-jail after (tail) ==="; tail -6 screen_after.txt
echo "=== plain after   (tail) ==="; tail -6 screen_after_plain.txt
