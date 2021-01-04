#NoEnv  ; Recommended for performance and compatibility with future AutoHotkey releases.
; #Warn  ; Enable warnings to assist with detecting common errors.
SendMode Input  ; Recommended for new scripts due to its superior speed and reliability.
SetWorkingDir %A_ScriptDir%  ; Ensures a consistent starting directory.


; Define these so we can print them anywhere
extracted := 0
last_mrn := "BEFORE SCRIPT VALUE"


!j::


; Script setup
; Set coordinates relative to screen
; This may allow the script to continue if small windows pop up
CoordMode, Mouse, Screen
CoordMode, Pixel, Screen
WinMaximize, ahk_exe MUSEEditor.exe
count := 0


; Open file to read input from, line by line
Loop, Read, C:\Users\MuseAdmin\Desktop\mrn_date.csv
{
    mrn_time := A_LoopReadLine
    mrn_time := StrSplit(mrn_time, ",")
    mrn := mrn_time[1]
    time := mrn_time[2]
    time := StrSplit(StrSplit(time, A_Space)[1], "-")
    year := time[1]
    month := time[2]
    day := time[3]

    ; Switch to MUSE Editor
    WinActivate, ahk_exe MUSEEditor.exe

    ; Remove previous input from search box
    Loop, 20 {
       Click, 139, 819
       Send, {Delete}
       Send, {Backspace}
    }

    ; Enter new input to search box
    Click, 139, 819
    Send, %mrn%

    ; Click date field dots -> on date
    Click, 238, 918
    Click, 235, 802

    ; Click day and type date
    Click, 141, 919
    Send, %day%

    ; Click month and type month
    Click, 163, 919
    Send, %month%

    ; Click year and type year
    Click, 191, 919
    Send, %year%


    ; SCRIPT OTHER SEARCH CRITERIA HERE


    ; Click search button
    Click, 73, 998

    ; Select all items in search result list
    Click, 91, 31
    Click, 110, 106

    ; Print list
    Click, 415, 59
    Sleep, 750
    Click, 1192,800
    extracted++
    last_mrn := mrn

    ; Wait for MUSE Editor to unfreeze
    Loop {
        ; Search for a black pixel in the space where window goes blank/freezes
        PixelSearch, Px, Py, 478, 242, 1379, 719, 0x000000, Fast
        If Not ErrorLevel {
            Break
        }
        Sleep, 100
    }

    ; Reset MUSE Editor after 1000 searches
    count++
    If (count > 1000)
    {
        Sleep, 20000
        WinClose, ahk_exe MUSEEditor.exe
        Run "C:\Program Files (x86)\MUSE\MUSEEditor.exe", C:\Program Files (x86)\MUSE
        Sleep, 1000
        count := 0
        Sleep, 20000
    }
}

MsgBox, Extracted %extracted% MRNs
Return


!r::
MsgBox, Script Reloaded
Reload
Return


!Esc::
MsgBox, Halted after %extracted% searches. Last MRN was %last_mrn%
ExitApp
