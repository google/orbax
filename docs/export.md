# Orbax Export

[SavedModel](https://www.tensorflow.org/guide/saved_model) format.

## Exporting

### API Overview

Orbax Export provides three classes.

-   [`JaxModule`](https://github.com/google/orbax/tree/main/orbax/export/jax_module.py)
    wraps a JAX function and its parameters to an exportable and callable
    closure.
-   [`ServingConfig`](https://github.com/google/orbax/tree/main/orbax/export/export_manager.py;l=12)
    defines a serving configuration for a `JaxModule`, including
    [a signature key and an input signature][1], and optionally pre- and
    post-processing functions and extra [TrackableResources][2].
-   [`ExportManager`](https://github.com/google/orbax/tree/main/orbax/export/export_manager.py;l=35)
    builds the actual [serving signatures][1] based on a `JaxModule` and a list
    of `ServingConfig`s, and saves them to the SavedModel format.

### Simple Example Usage

#### Setup

```python
# Import Orbax Export classes.
from orbax.export import ExportManager
from orbax.export import JaxModule
from orbax.export import ServingConfig

# Prepare the parameters and model function to export.
params = ... # A pytree of the JAX model parameters.

def model_fn(params, inputs):  # The JAX model function to export.
  ...

def preprocess(*inputs):  # Optional: preprocessor in TF.
  ...

def postprocess(model_fn_outputs):  # Optional: post-processor in TF.
  ...

```

#### Exporting a JAX model to a CPU SavedModel

```python
# Construct a JaxModule where JAX->TF conversion happens.
jax_module = JaxModule(params, model_fn)
# Export the JaxModule along with one or more serving configs.
export_mgr = ExportManager(
  jax_module, [
    ServingConfig(
      'serving_default',
      input_signature=[tf.TensorSpec(...)],
      tf_preprocessor=preprocess,
      tf_postprocessor=postprocess
    ),
])
export_mgr.save(output_dir)
```


### Known issues

## Error message "JaxModule only take single arg as the input".

Orbax is designed to take a JAX Module in the format of a Callable with
parameters of type PyTree and model inputs of type PyTree. If your JAX function
takes multiple inputs, you must pack them into a single JAX PyTree. Otherwise,
you will encounter this error message.

To solve this problem, you can update the `ServingConfig.tf_preprocessor`
function to pack the inputs into a single JAX PyTree. For example, our model
takes two inputs `x` and `y`. You can define the `ServingConfig.tf_preprocessor`
pack them into a list `[x, y]`.

```python
def tf_preprocessor(x, y):
  # put the normal tf_preprocessor codes here.
  return [x, y] # pack it into a single list for jax model_func.

jax_module = orbax.export.JaxModule(params, model_func)
export_mgr = orbax.export.ExportManager(
  jax_module,
  [
      orbax.export.ServingConfig(
          'serving_default',
          input_signature=[tf.TensorSpec([16]), tf.TensorSpec([16])],
          tf_preprocessor=tf_preprocessor,
      )
  ],
)
export_mgr.save('/tmp/foo')
```

## Validating

### API Overview

library that can be used to validate the JAX model and its exported TF
[SavedModel](https://www.tensorflow.org/guide/saved_model) format.

Users must finish the JAX model exporting first. Users can export the model by
orbax.export or manually.

Orbax.export.validate provides those classes:

*   [`ValidationJob`](https://github.com/google/orbax/tree/main/orbax/export/validate/validation_job.py)
    take the model and data as input, then output the result.
*   [`ValidationReport`](https://github.com/google/orbax/tree/main/orbax/export/validate/validation_report.py)
    compare the JAX model and TF SavedModel results, then generate the formatted
    report.
*   [`ValidationManager`](https://github.com/google/orbax/tree/main/orbax/export/validate/validation_manager.py)
    take `JaxModule` as inputs and wrap the validation e2e flow.

### Simple Example Usage

Here is a simple example.

```python
from orbax.export.validate import ValidationManager
from orbax.export import JaxModule
from orbax.export import ServingConfig

params = {'bias': jnp.array(1)}
apply_fn = lambda params, inputs: inputs['x'] + params['bias']
jax_module = JaxModule(params, apply_fn)
batch_inputs = [{'x': np.arange(8).astype(np.int32)}] * 16

serving_configs = [
  ServingConfig(
                  'serving_default',
                  input_signature=[{
                    'x': tf.TensorSpec((), tf.dtypes.int32, name='x')
                    }],
  ),
]
validation_mgr = ValidationManager(jax_module.jax_methods, serving_configs,
                                       batch_inputs)

tf_saved_model_path = ...
loaded_model = tf.saved_model.load(tf_saved_model_path)
# for tpu model:
# loaded_model = tf.saved_model.load(tf_saved_model_path, tags=['serve', 'tpu'])
validation_reports = validation_mgr.validate(loaded_model)
```

`validation_reports` is a python dict and the key is TF SavedModel serving_key.

```python
for key in validation_reports:
  self.assertEqual(validation_reports[key].status.name,'Pass')
  # Users can also save the converted json to file.
  json_str = validation_reports[key].to_json()
```

### Limitation

Here we list those limitation of Orbax.export validate module.

*   To avoid ambiguity, the model inputs must be a dictionary from string to
    Tensor. The input signature TensorSpec name should be same as the key. This
    is an example.

```
python image = random.normal(random.PRNGKey(1), (32, 28, 28, 1),
dtype=jnp.float32) label = random.randint(random.PRNGKey(1), (32,), 0, 10,
dtype=jnp.int64) mnist_inputs = {'image': image, 'label': label}

mnist_input_signature = [ { 'image': TensorSpec(shape=(32, 28, 28, 1),
dtype=tf.float32, name='image'), 'label': TensorSpec(shape=(32,),
dtype=tf.int32, name='label') } ]
```

*   Because the TF SavedModel the returned object is always a map. If the jax
    model output is a sequence, TF SavedModel will convert it to map. The tensor
    names are fairly generic, like output_0. To help `ValidationReport` module
    can do apple-to-apple comparison between JAX model and TF model result, we
    suggest users modify the model output as a dictionary.

## Examples

Check-out the [examples](https://github.com/google/orbax/tree/main/orbax/export/examples)
directory for a number of examples using Orbax Export.

[1]: https://www.tensorflow.org/guide/saved_model#specifying_signatures_during_export
[2]: https://www.tensorflow.org/api_docs/python/tf/saved_model/experimental/TrackableResource
