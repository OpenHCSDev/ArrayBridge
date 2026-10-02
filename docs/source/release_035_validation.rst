ArrayBridge 0.3.5 release preparation
===================================

Parent integration owner; base ea3f2a4cc91c4810d12343f58f85c1195e1a41e6.
The reviewed callable-native dtype default and durable decorator-context
changes are already merged through PR2 and PR3. This release preparation changes
only the existing package version authority from 0.3.4 to 0.3.5 and records
validation. It adds no implementation, compatibility API or parallel registry.

OpenHCS PR217 requires ``dtype_config_default``. A fresh PyPI ArrayBridge0.3.4
wheel raises TypeError at its BaSiC adapter's decoration. Its recorded local
source candidate has that API, but is not a public release. OpenHCS now requires
ArrayBridge>=0.3.5,<0.4 so ordinary installation cannot select the old API.

Persistent evidence::

    /home/ts/wt/openhcs-issue-batch-20260929/artifact-parent-20261001

Focused provider-free checks, using the existing Python3.12 environment::

    PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest \
      -o addopts='' -q tests/test_callable_dtype_default.py \
      tests/test_durable_decorator_context.py

All seven tests pass in 0.12 seconds. They retain native floating-point values,
explicit runtime dtype overrides, unchanged default preservation, rejection of
untyped declarations and durable serialization without thread-local leakage.
The original complete-suite acceptance of the reviewed feature PRs is separate;
these seven checks do not claim a full framework/GPU matrix.

The wheel was built through ``uv build --offline --wheel`` from this source
using cached build requirements. Twine6.2 accepts the resulting metadata2.4
wheel. An initial no-isolation build lacked Hatchling in the installed OpenHCS
environment and failed; it did not install packages or count as a pass.

A fresh process loaded that exact 0.3.5 wheel archive ahead of the actual
installed OpenHCS candidate and executed the real BaSiC adapter on 24 synthetic
32x32 SITE observations. It returned finite float32 correction with fractional
pixels, preserved source pixels and matched the same-fit correction formula.
Flatfield RMSE against the known synthetic shading field is 0.0007167315491296268.
``arraybridge-035-installed-fit.log`` records the actual wheel and site-packages
import paths. This establishes local wheel/API interoperability, not registry
resolution, a full installed MCP rerun or biological accuracy. The separate
installed MCP journey is retained with OpenHCS PR217.

Publication is not performed by this PR. The parent requested separate owner
approval for the companion ArrayBridge0.3.5 publication. No tag was created,
workflow dispatched, registry upload performed or release availability claimed.
After approval, merge normally, tag the reviewed main version and use the
existing trusted publishing workflow. Verify the actual registry wheel before
claiming ordinary OpenHCS dependency installation.
