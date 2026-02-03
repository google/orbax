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
        if name and tags:
          yield name, tags


def run(root_dir, output_file):
  """Runs the script to generate tagged tests file."""
  tests_by_tag = {tag: [] for tag in TAG_MAPPING.values()}

  count = 0
  for dirpath, _, filenames in os.walk(root_dir):
    if 'BUILD' in filenames:
      count += 1
      build_file = os.path.join(dirpath, 'BUILD')
      package_path = '//' + dirpath
      for name, tags in get_build_targets(build_file):
        for tag in tags:
          if tag in TAG_MAPPING:
            tests_by_tag[TAG_MAPPING[tag]].append(f'{package_path}:{name}')

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
  orbax_dir = 'orbax/checkpoint'
  output = 'orbax/checkpoint/_src/testing/multiprocess_unit_test/build_tags/tagged_tests.yaml'
  run(orbax_dir, output)
