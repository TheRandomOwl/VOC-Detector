#define MyAppVersion "0.7.0"

[Setup]
AppName=VOC GUI
AppVersion={#MyAppVersion}
AppPublisherURL="https://github.com/TheRandomOwl/VOC-Detector"
AppUpdatesURL="https://github.com/TheRandomOwl/VOC-Detector/releases"
AppPublisher=Nathan Perry
AppSupportURL="https://github.com/TheRandomOwl/VOC-Detector/blob/main/README.md"
DefaultDirName={autopf}\VOC GUI
DefaultGroupName=VOC GUI
OutputBaseFilename=VOC GUI Installer {#MyAppVersion}
Compression=lzma
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "dist\*.*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\VOC GUI"; Filename: "{app}\voc-gui\voc-gui.exe"
Name: "{commondesktop}\VOC GUI"; Filename: "{app}\voc-gui\voc-gui.exe"

[Run]
Filename: "{app}\voc-gui\voc-gui.exe"; Description: "Launch VOC GUI"; Flags: nowait postinstall skipifsilent