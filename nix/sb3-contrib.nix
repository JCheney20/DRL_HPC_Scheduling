{
  buildPythonPackage,
  fetchPypi,
  stable-baselines3
}:
buildPythonPackage rec {
  pname = "sb3-contrib";
  version = "2.6.0";
  format = "wheel";
  dontUseNinjaBuild = true;
  src = fetchPypi {
    inherit version format;
    pname = "sb3_contrib";
    dist = "py3";
    python = "py3";
    hash = "sha256-dEDSpH3NKZbG9EKGoFkkX2JDPSVllRxEVnPGqDdC7MI=";
  };

  propagatedBuildInputs = [
    stable-baselines3
  ];
}
