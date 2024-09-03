[Setup]
AppName=VOC GUI
AppVersion=0.5.2
DefaultDirName={autopf}\VOC GUI
DefaultGroupName=VOC GUI
OutputBaseFilename=VOC GUI Installer
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\*.*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\VOC GUI"; Filename: "{app}\voc-gui\voc-gui.exe"
Name: "{commondesktop}\VOC GUI"; Filename: "{app}\voc-gui\voc-gui.exe"

[Run]
Filename: "{app}\voc-gui\voc-gui.exe"; Description: "Launch VOC GUI"; Flags: nowait postinstall skipifsilent