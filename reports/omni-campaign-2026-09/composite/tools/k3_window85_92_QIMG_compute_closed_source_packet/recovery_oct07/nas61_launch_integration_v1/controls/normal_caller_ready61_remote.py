"""Read the existing caller's actual startup line before opening the boot barrier."""
import datetime, json, pathlib, time
p = pathlib.Path(I['output_root']) / 'capacity_host_caller.log'
pid = FREEZE['host_caller_pid']
assert pathlib.Path('/proc', str(pid)).is_dir(), 'Actual capacity caller exited before serving'
cmd = pathlib.Path('/proc', str(pid), 'cmdline').read_bytes().replace(b'\0', b' ').decode()
assert I['runtime_root'] + '/production_control/k3_capacity_host_loop_v1.py' in cmd
assert I['runtime_root'] + '/production_control/hostcapacity_caller_config.json' in cmd
deadline = time.monotonic() + 10
while True:
    assert pathlib.Path('/proc', str(pid)).is_dir(), 'Actual capacity caller exited before serving'
    if p.is_file():
        assert not p.is_symlink() and p.stat().st_size <= 1 << 20
        rows = p.read_text().splitlines()
        for line in rows:
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if data.get('serving') is True:
                assert data['run_uuid'] == I['run_uuid']
                assert data['requests'] == I['output_root'] + '/capacity_requests'
                assert data['responses'] == I['output_root'] + '/capacity_responses'
                print(json.dumps({'ok': True, 'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  'host_caller_pid': pid, 'actual_serving_line': data}))
                raise SystemExit(0)
    assert time.monotonic() < deadline, 'Actual caller serving line not observed in bounded startup check'
    time.sleep(0.2)
