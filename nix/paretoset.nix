{
  buildPythonPackage,
  fetchPypi,
}:
buildPythonPackage rec {
  pname = "paretoset";
  version = "1.2.5";
  format = "wheel";
  dontUseNinjaBuild = true;
  src = fetchPypi {
    inherit version format;
    pname = "paretoset";
    dist = "py3";
    python = "py3";
    hash = "sha256-43Yxh7vJvMf+6u25/MY9YCbOLGFL+EaP8jNK65bPJuk=";
  };

}
