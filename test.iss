[Setup]
AppName=My CLI Application
AppVersion=1.0
DefaultDirName={autopf}\MyCLIApp
DefaultGroupName=My CLI Application
OutputBaseFilename=MyCLIAppSetup
Compression=lzma
SolidCompression=yes
PrivilegesRequired=lowest

[Files]
Source: "dist\voc-cli.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\My CLI Application"; Filename: "{app}\voc-cli.exe"

[Run]
Filename: "{app}\voc-cli.exe"; Description: "Launch My CLI Application"; Flags: nowait postinstall skipifsilent

[Registry]
; Add the installation directory to the PATH environment variable
;Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: string; ValueName: "Path"; ValueData: "{app}"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "Path"; ValueData: "{olddata};{app}"; Flags: uninsdeletevalue
[UninstallDelete]
Type: files; Name: "{app}\voc-cli.exe"