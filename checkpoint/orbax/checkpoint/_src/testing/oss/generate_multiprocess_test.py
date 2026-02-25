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
    'orbax/checkpoint/experimental',
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


def inherits_from_multiprocess_test(test_file_path):
  """Checks if test file inherits from MultiProcessTest."""
  try:
    with open(test_file_path, 'r') as f:
      content = f.read()
  except FileNotFoundError:
    return False
  try:
    tree = ast.parse(content, filename=test_file_path)
  except SyntaxError:
    return False

  imported_as_name = None  # if imported as `from ... import MultiProcessTest`
  imported_as_module = []  # if imported as `from ... import multiprocess_test`

  for node in tree.body:
    if isinstance(node, ast.ImportFrom):
      if node.module == 'orbax.checkpoint._src.testing.multiprocess_test':
        for alias in node.names:
          if alias.name == 'MultiProcessTest':
            imported_as_name = alias.asname or alias.name
      elif node.module == 'orbax.checkpoint._src.testing':
        for alias in node.names:
          if alias.name == 'multiprocess_test':
            imported_as_module.append(alias.asname or alias.name)

  if not imported_as_name and not imported_as_module:
    return False

  for node in tree.body:
    if isinstance(node, ast.ClassDef):
      for base in node.bases:
        if (
            imported_as_name
            and isinstance(base, ast.Name)
            and base.id == imported_as_name
        ):
          return True
        if (
            imported_as_module
            and isinstance(base, ast.Attribute)
            and isinstance(base.value, ast.Name)
            and base.value.id in imported_as_module
            and base.attr == 'MultiProcessTest'
        ):
          return True
  return False


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
      if 'name' in kwargs and 'tags' in kwargs:
        name = get_str_val(kwargs['name'])
        tags = get_list_val(kwargs['tags'])
        srcs = get_list_val(kwargs['srcs']) if 'srcs' in kwargs else []
        if name and tags:
          yield name, tags, srcs


def run(root_dir, output_file):
  """Runs the script to generate tagged tests file."""
  tests_by_tag = {tag: [] for tag in TAG_MAPPING.values()}
  tests_by_tag['processes:1'] = []

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
      for name, tags, srcs in get_build_targets(build_file):
        if srcs and any(
            os.path.join(dirpath, srcs[0]).startswith(p) for p in EXCLUDED_PATHS
        ):
          continue
        is_multiprocess = False
        if srcs:
          is_multiprocess = inherits_from_multiprocess_test(
              os.path.join(dirpath, srcs[0])
          )
        target_path = f'{package_path}:{name}'
        if not is_multiprocess:
          tests_by_tag['processes:1'].append(target_path)
        else:
          for tag in tags:
            if tag in TAG_MAPPING:
              tests_by_tag[TAG_MAPPING[tag]].append(target_path)

  print(f'Processed {count} BUILD files.')

  for tag in tests_by_tag:
    tests_by_tag[tag] = sorted(list(set(tests_by_tag[tag])))

  header = """# DO NOT EDIT!
"""
  os.makedirs(os.path.dirname(output_file), exist_ok=True)
  with open(output_file, 'w') as f:
    f.write(header)
    yaml.dump(tests_by_tag, f, default_flow_style=False)
  print(f'Output written to {output_file}')


if __name__ == '__main__':
  if 'BUILD_WORKING_DIRECTORY' in os.environ:
    os.chdir(os.environ['BUILD_WORKING_DIRECTORY'])
  orbax_dir = 'orbax/checkpoint'
  output = 'orbax/checkpoint/_src/testing/oss/tagged_tests.yaml'
  run(orbax_dir, output)
