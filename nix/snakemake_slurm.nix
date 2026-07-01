{
  buildPythonPackage,
  fetchPypi,
}:
buildPythonPackage rec {
  pname = "snakemake-executor-plugin-slurm";
  version = "2.7.1";
  format = "wheel";
  dontUseNinjaBuild = true;
  src = fetchPypi {
    inherit version format;
    pname = "snakemake_executor_plugin_slurm";
    dist = "py3";
    python = "py3";
    hash = "sha256-TGMha9t8AWt3THfFjcZj2B0vyldWokei4au0dr8HCPE=";
  };

}
