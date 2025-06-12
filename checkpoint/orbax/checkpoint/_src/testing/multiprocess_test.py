# Copyright 2025 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper for running multi-process tests."""

import os
import pathlib
import re
import signal
import subprocess
import time

from absl import app
from absl import flags
from absl.testing import absltest
import jax
from jax import config
from orbax.checkpoint._src.multihost import multihost
import portpicker

NUM_PROCESSES = flags.DEFINE_integer(
    "num_processes", None, "Number of processes to use."
)

_GPUS_PER_PROCESS = flags.DEFINE_integer(
    "gpus_per_process",
    0,
    "Number of GPUs per worker process.",
)

TPU_CHIPS_PER_PROCESS = flags.DEFINE_integer(
    "tpu_chips_per_process",
    0,
    "Number of TPU chips per worker process.",
)

_WORKER_SHUTDOWN_TIMEOUT = flags.DEFINE_integer(
    "worker_shutdown_timeout",
    15,
    "JAX shutdown timeout duration in seconds for each subprocess worker. If "
    "your test is timing out, try increasing this value.",
)

_EXTRA_TEST_ARGS = flags.DEFINE_multi_string(
    "extra_test_args", [], "Extra flags to pass to worker process."
)

# For internal use.
MULTIPROCESS_TEST_WORKER_ID = flags.DEFINE_integer(
    "multiprocess_test_worker_id",
    -1,
    "TPU worker id. Set by main test process; should not be set by users.",
)

expect_failures_with_regex = None


def main():
  # We don't call googletest.main() because if we are the main process we
  # don't want to run the tests; instead we want to fork worker processes that
  # do. However, the googletest framework would otherwise report an error in the
  # main process if googletest.main() is not called, which the following line
  # suppresses.
  config.config_with_absl()
  app.run(_main)


class GracefulKiller:
  """Add a signal handler that sets a flag if SIGINT or SIGTERM are caught."""

  # From https://stackoverflow.com/a/31464349
  kill_now = False

  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, sig_num, unused_stack_frame):
    print(f"Caught signal: {signal.Signals(sig_num).name} ({sig_num})")
    self.kill_now = True


def _main(argv):
  if NUM_PROCESSES.value is None:
    raise ValueError("--num_processes must be specified.")

  if MULTIPROCESS_TEST_WORKER_ID.value >= 0:
    absltest.run_tests(argv, [], {})
    return

  num_processes = NUM_PROCESSES.value
  gpus_per_process = _GPUS_PER_PROCESS.value
  tpu_chips_per_process = TPU_CHIPS_PER_PROCESS.value
  num_tpu_chips = num_processes * tpu_chips_per_process
  if num_tpu_chips == 0:
    pass
  elif num_tpu_chips == 1:
    assert tpu_chips_per_process == 1
    deepsea_host_bounds = "1,1,1"
    deepsea_chips_per_host_bounds = "1,1,1"
  elif num_tpu_chips == 4:
    if tpu_chips_per_process == 1:
      deepsea_host_bounds = "2,2,1"
      deepsea_chips_per_host_bounds = "1,1,1"
    elif tpu_chips_per_process == 2:
      deepsea_host_bounds = "2,1,1"
      deepsea_chips_per_host_bounds = "1,2,1"
    elif tpu_chips_per_process == 4:
      deepsea_host_bounds = "1,1,1"
      deepsea_chips_per_host_bounds = "2,2,1"
    else:
      raise ValueError(
          "Invalid number of TPU chips per worker {}".format(
              tpu_chips_per_process
          )
      )
  elif num_tpu_chips == 8:
    if tpu_chips_per_process == 1:
      deepsea_host_bounds = "4,2,1"
      deepsea_chips_per_host_bounds = "1,1,1"
    else:
      # TODO(phawkins): implement other cases.
      raise ValueError(
          "Invalid number of TPU chips per worker {}".format(
              tpu_chips_per_process
          )
      )
  else:
    raise ValueError(f"Invalid number of TPU chips {num_tpu_chips}")

  slicebuilder_ports = [
      portpicker.pick_unused_port() for _ in range(num_processes)
  ]
  slicebuilder_addresses = ",".join(
      f"localhost:{port}" for port in slicebuilder_ports
  )
  jax_port = portpicker.pick_unused_port()

  subprocesses = []
  output_filenames = []
  output_files = []
  for i in range(num_processes):
    env = os.environ.copy()

    args = [
        "/proc/self/exe",
        f"--multiprocess_test_worker_id={i}",
        "--logtostderr",
        f"--num_processes={num_processes}",
        f"--jax_num_tasks={num_processes}",
        f"--jax_task_id={i}",
        f"--jax_controller_address=localhost:{jax_port}",
        "--jax_heartbeat_timeout=3s",
        "--jax_heartbeat_interval=1s",
        "--jax_max_missing_heartbeats=3",
        f"--jax_distributed_shutdown_timeout={_WORKER_SHUTDOWN_TIMEOUT.value}s",
        "--vmodule=client=10,service=10",
    ]
    if i == 0:
      args += [f"--jax_port={jax_port}"]

    if num_tpu_chips > 0:
      # We must set a CLOUD_TPU_TASK_ID, otherwise the TPU runtime expects to
      # be able to look up the Borg task number from the Borglet info.
      env["CLOUD_TPU_TASK_ID"] = str(i)
      chips = list(
          range(
              i * tpu_chips_per_process,
              (i + 1) * tpu_chips_per_process,
          )
      )
      excluded_chips = [d for d in range(num_tpu_chips) if d not in chips]
      args += [
          "--deepsea_hal_excluded_devs={}".format(
              ",".join(map(str, excluded_chips))
          ),
          f"--deepsea_host_bounds={deepsea_host_bounds}",
          f"--deepsea_chips_per_host_bounds={deepsea_chips_per_host_bounds}",
          f"--deepsea_slice_builder_worker_addresses={slicebuilder_addresses}",
          f"--deepsea_slice_builder_worker_port={slicebuilder_ports[i]}",
          # --jax_allow_unused_tpus suppresses a check in JAX that verifies the
          # number of JAX TPU devices found is equal to the hardware devices
          # that exist.
          "--jax_allow_unused_tpus",
      ]

    if gpus_per_process > 0:
      gpus = range(i * gpus_per_process, (i + 1) * gpus_per_process)
      env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))

    args += _EXTRA_TEST_ARGS.value

    undeclared_outputs = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp")
    stdout_name = f"{undeclared_outputs}/jax_{i}_stdout.log"
    stderr_name = f"{undeclared_outputs}/jax_{i}_stderr.log"
    stdout = open(stdout_name, "wb")
    stderr = open(stderr_name, "wb")
    print(f"Launching process {i}:")
    print(f"  stdout: {stdout_name}")
    print(f"  stderr: {stderr_name}")
    proc = subprocess.Popen(args, env=env, stdout=stdout, stderr=stderr)
    subprocesses.append(proc)
    output_filenames.append((stdout_name, stderr_name))
    output_files.append((stdout, stderr))

  print(" All launched, running ".center(80, "="), flush=True)

  # Wait for all the children to finish or for a SIGTERM from TAP. If we get
  # SIGTERM, we still want to collect their logs, so kill them and continue.
  killer = GracefulKiller()
  running_procs = {i: p for i, p in enumerate(subprocesses)}
  while not killer.kill_now and running_procs:
    time.sleep(0.1)
    for i, proc in list(running_procs.items()):
      if proc.poll() is not None:
        print(f"Process {i} finished.", flush=True)
        running_procs.pop(i)
  if killer.kill_now and running_procs:
    print("Caught termination, terminating remaining children.", flush=True)

    # Send a SIGTERM to each child process, to let it know it should terminate.
    for i, proc in running_procs.items():
      proc.terminate()
      print(f"Process {i} terminated.", flush=True)

    # On test timeout, Forge first sends a SIGTERM (a "soft" kill signal, that
    # the test can intercept, in order to do some cleanup, log flushing, etc).
    # After a grace period of 15 seconds, Forge sends a SIGKILL (a "hard" kill),
    # see http://yaqs/eng/q/4559876738121728#n4588728130600960. We give the
    # child process(es) a few seconds for their own cleanup, and keep the rest
    # (up to 15s) for copying the children logs into our own.
    time.sleep(5)

    # Send a SIGKILL (a "hard" kill) to each child process. This is CRITICAL:
    # without it, this process may end up waiting a long time on the proc.wait()
    # below, and never get to saving the children logs, making test timeouts
    # very hard to debug.
    for i, proc in running_procs.items():
      proc.kill()
      print(f"Process {i} killed.")
    print("Killed all child processes.", flush=True)

  retvals = []
  stdouts = []
  stderrs = []
  for proc, fds, (stdout, stderr) in zip(
      subprocesses, output_files, output_filenames
  ):
    retvals.append(proc.wait())
    for fd in fds:
      fd.close()
    stdouts.append(pathlib.Path(stdout).read_text(errors="replace"))
    stderrs.append(pathlib.Path(stderr).read_text(errors="replace"))

  print(" All finished ".center(80, "="), flush=True)

  print(" Summary ".center(80, "="))
  for i, (retval, stdout, stderr) in enumerate(zip(retvals, stdouts, stderrs)):
    m = re.search(r"Ran \d+ tests? in [\d.]+s\n\n.*", stderr, re.MULTILINE)
    result = m.group().replace("\n\n", "; ") if m else "Test crashed?"
    print(
        f"Process {i}, ret: {retval}, len(stdout): {len(stdout)}, "
        f"len(stderr): {len(stderr)}; {result}"
    )

  print(" Detailed logs ".center(80, "="))
  for i, (retval, stdout, stderr) in enumerate(zip(retvals, stdouts, stderrs)):
    print(f" Process {i}: return code: {retval} ".center(80, "="))
    if stdout:
      print(f" Process {i} stdout ".center(80, "-"))
      print(stdout)
    if stderr:
      print(f" Process {i} stderr ".center(80, "-"))
      print(stderr)

  print(" Done detailed logs ".center(80, "="), flush=True)
  for i, (retval, stderr) in enumerate(zip(retvals, stderrs)):
    if retval != 0:
      if expect_failures_with_regex is not None:
        assert re.search(
            expect_failures_with_regex, stderr
        ), f"process {i} failed, expected regex: {expect_failures_with_regex}"
      else:
        assert retval == 0, f"process {i} failed, return value: {retval}"


class MultiProcessTest(absltest.TestCase):
  # TODO(b/378138653) Support TPUless MultiProcessTest.

  def setUp(self):
    """Start distributed service."""
    super().setUp()
    assert jax.process_count() == NUM_PROCESSES.value, (
        jax.process_count(),
        NUM_PROCESSES.value,
    )
    # Make sure all processes are at the same test case.
    client = multihost.get_jax_distributed_client()
    # Note that the name of this barrier is long and complicated, to prevent
    # any collisions with barriers in user test code.
    client.wait_at_barrier(
        f"multiprocess_test_ensure_all_processes_arrive_at_test_case_{self._testMethodName}",
        10000,
    )

  def multiprocess_create_tempdir(
      self, name: str | None = None
  ) -> str:
    """Creates a temporary directory for the test."""
    directory = self._get_tempdir_path_test()
    if name is not None:
      directory = os.path.join(directory, name)
    if jax.process_index() == 0:
      os.makedirs(directory, exist_ok=False)
    jax.experimental.multihost_utils.sync_global_devices(
        "multiprocess_create_tempdir"
    )
    return directory
