let
    pkgs = import <nixpkgs> {};
    stdenv = pkgs.stdenv;
    pypkgs = pkgs.python310Packages;
    czifile = pypkgs.buildPythonPackage rec {
      pname = "czifile";
      version = "2019.7.2";
      src = pypkgs.fetchPypi {
        inherit pname version;
        sha256 = "sha256-BMDmvtOyTRv0K8LPiZpaCJhmQTeTBc6IYA/RxxBIZDY=";
      };
      doCheck = false;
      propagatedBuildInputs = [
        pypkgs.numpy
        pypkgs.tifffile
      ];
    };

in pkgs.mkShell rec {
    buildInputs = [
        pkgs.python3
        pypkgs.jupyter
        pypkgs.numpy
        pypkgs.matplotlib
        pypkgs.scipy
        pypkgs.scikitimage
        pypkgs.opencv4
        pypkgs.pandas
        pypkgs.xmltodict
        pypkgs.doit
        czifile
    ];
    QT_QPA_PLATFORM_PLUGIN_PATH = "${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins/platforms";
}
