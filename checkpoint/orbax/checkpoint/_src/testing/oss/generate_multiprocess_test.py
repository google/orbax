# Copyright 2026 The Orbax Authors.
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

"""Script to generate YAML file with test targets based on tags."""

import ast
import collections
import os
import sys

import yaml


TAG_MAPPING = {
}
TEST_RULES = [
    'py_strict_test',
    'pytype_strict_test',
    'pytype_strict_contrib_test',
]
EXCLUDED_PATHS = [
    'orbax/checkpoint/experimental/model_surgery',
    'orbax/checkpoint/experimental/v1',
    'orbax/checkpoint/experimental/emergency/p2p',
    'orbax/checkpoint/experimental/emergency/checkpoint_manager_test.py',
    'orbax/checkpoint/experimental/emergency/replicator_checkpoint_manager_test.py',
    'orbax/checkpoint/google',
]


def get_kwargs(call_node):
  """Returns kwargs of a call node."""
  kwargs = {}
  for keyword in call_node.keywords:
    kwargs[keyword.arg] = keyword.value
  return kwargs


def get_list_val(node):
  """Returns list of strings from an AST list node."""
  if isinstance(node, ast.List):
    result = []
    for elt in node.elts:
      if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
        result.append(elt.value)
    return result
  return []


def get_str_val(node):
  """Returns string value from an AST node."""
  if isinstance(node, ast.Constant) and isinstance(node.value, str):
    return node.value
  return None


def get_num_processes(args):
  """Returns num_processes from args."""
  for arg in args:
    if arg.startswith('--num_processes='):
      try:
        return int(arg.split('=', 1)[1])
      except ValueError:
        return None
  return None


def get_build_targets(build_file_path):
  """Yields test targets and tags from a BUILD file."""
  try:
    with open(build_file_path, 'r') as f:
      content = f.read()
  except FileNotFoundError:
    return
  try:
    tree = ast.parse(content, filename=build_file_path)
  except SyntaxError as e:
    print(f'Could not parse {build_file_path}: {e}', file=sys.stderr)
    return

  for node in tree.body:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
      continue
    call = node.value

    rule_name = ''
    if isinstance(call.func, ast.Name):
      rule_name = call.func.id
    elif isinstance(call.func, ast.Attribute):
      rule_name = call.func.attr

    if rule_name in TEST_RULES:
      kwargs = get_kwargs(call)
      if 'name' in kwargs:
        name = get_str_val(kwargs['name'])
        tags = get_list_val(kwargs['tags']) if 'tags' in kwargs else []
        srcs = get_list_val(kwargs['srcs']) if 'srcs' in kwargs else []
        args = get_list_val(kwargs['args']) if 'args' in kwargs else []
        if name:
          yield name, tags, srcs, args


def run(root_dir, output_file):
  """Runs the script to generate tagged tests file."""
  tests_by_tag = collections.defaultdict(list)

  count = 0
  for dirpath, dirnames, filenames in os.walk(root_dir):
    if any(dirpath.startswith(p) for p in EXCLUDED_PATHS):
      dirnames[:] = []
      continue

    original_dirs = list(dirnames)
    dirnames[:] = []
    for d in original_dirs:
      if not any(
          os.path.join(dirpath, d).startswith(p) for p in EXCLUDED_PATHS
      ):
        dirnames.append(d)

    if 'BUILD' in filenames:
      count += 1
      build_file = os.path.join(dirpath, 'BUILD')
      package_path = dirpath.removeprefix('third_party/py/')
      for name, tags, srcs, args in get_build_targets(build_file):
        if not any(tag in TAG_MAPPING for tag in tags):
          continue
        if srcs and any(
            os.path.join(dirpath, srcs[0]).startswith(p) for p in EXCLUDED_PATHS
        ):
          continue
        target_path = f'{package_path}:{name}'
        num_processes = get_num_processes(args)
        if num_processes and num_processes > 1:
          tag = f'processes:{num_processes}'
          tests_by_tag[tag].append(target_path)
        else:
          tests_by_tag['processes:1'].append(target_path)

  print(f'Processed {count} BUILD files.')

  result_dict = {}
  for tag in tests_by_tag:
    result_dict[tag] = sorted(list(set(tests_by_tag[tag])))

  header = """# DO NOT EDIT!
"""
  os.makedirs(os.path.dirname(output_file), exist_ok=True)
  with open(output_file, 'w') as f:
    f.write(header)
    yaml.dump(result_dict, f, default_flow_style=False)
  print(f'Output written to {output_file}')


if __name__ == '__main__':
  if 'BUILD_WORKING_DIRECTORY' in os.environ:
    os.chdir(os.environ['BUILD_WORKING_DIRECTORY'])
  orbax_dir = 'orbax/checkpoint'
  output = 'orbax/checkpoint/_src/testing/oss/tagged_tests.yaml'
  run(orbax_dir, output)
