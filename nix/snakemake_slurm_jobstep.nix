{
  buildPythonPackage,
  fetchPypi,
}:
buildPythonPackage rec {
  pname = "snakemake-executor-plugin-slurm-jobstep";
  version = "0.6.1";
  format = "wheel";
  dontUseNinjaBuild = true;
  src = fetchPypi {
    inherit version format;
    pname = "snakemake_executor_plugin_slurm_jobstep";
    dist = "py3";
    python = "py3";
    hash = "sha256-Eg4SLuExcle9lgiipFhSetn+RSCqGoYQnpzirDIjIe8=";
  };

  # The plugin's inner srun retries step creation forever ("step creation
  # still disabled, retrying (Requested nodes are busy)"), burning the whole
  # runtime ceiling when a node is wedged. --overlap lets the step coexist
  # with the extern/batch step; --immediate=300 fails the job after 5 min so
  # Snakemake's retries (profiles/slurm/config.yaml) resubmit it instead of
  # hanging for 14 h. Wheel install, so patch the installed file; the grep
  # guard makes a future version bump that changes the string fail the build
  # loudly instead of silently dropping the patch.
  postInstall = ''
    target=$out/lib/python*/site-packages/snakemake_executor_plugin_slurm_jobstep/__init__.py
    grep -q 'srun -n1 --cpu-bind=q' $target
    sed -i 's|srun -n1 --cpu-bind=q|srun -n1 --overlap --immediate=300 --cpu-bind=q|' $target
  '';
}
