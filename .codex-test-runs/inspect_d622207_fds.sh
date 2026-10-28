for p in 559294 559295; do
  echo "PID $p"
  if [ -d "/proc/$p/fd" ]; then
    for f in /proc/$p/fd/*; do
      readlink "$f" 2>/dev/null || true
    done | grep -E '/home/cereal/model-surgery-lab|\.safetensors|\.npy|\.npz|\.json|\.log' | sort -u | head -100
    echo IO
    cat "/proc/$p/io" 2>/dev/null || true
    echo WCHAN
    cat "/proc/$p/wchan" 2>/dev/null || true
  else
    echo EXITED
  fi
done
