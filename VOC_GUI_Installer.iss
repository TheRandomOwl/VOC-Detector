[Setup]
AppName=VOC GUI
AppVersion=1.0
DefaultDirName={autopf}\VOC GUI
DefaultGroupName=VOC GUI
OutputBaseFilename=VOC_GUI_Installer
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\voc-gui.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\voc-cli.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\VOC GUI"; Filename: "{app}\voc-gui.exe"
Name: "{commondesktop}\VOC GUI"; Filename: "{app}\voc-gui.exe"

[Run]
Filename: "{app}\voc-gui.exe"; Description: "Launch VOC GUI"; Flags: nowait postinstall skipifsilent