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

}
