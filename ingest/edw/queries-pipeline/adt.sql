SELECT distinct
    t3.PatientIdentityID as MRN
    ,t1.PatientEncounterID
    ,t1.RoomID
    ,t5.RoomNM
    ,t1.BedID
    ,t6.BedLabelNM
    ,t4.HospitalAdmitDTS
    ,t4.HospitalDischargeDTS
    ,t2.EffectiveDTS as TransferInDTS
    ,t1.EffectiveDTS as TransferOutDTS
    ,t1.ADTEventTypeCD
    ,t1.ADTEventTypeDSC
    ,t1.ADTEventSubtypeCD
    ,t1.ADTEventSubtypeDSC
    ,t1.DepartmentID
    ,t1.DepartmentDSC
    ,t1.EffectiveDTS
    ,t1.EventDTS
    ,t1.PatientClassCD
    ,t1.PatientClassDSC
    ,t1.PatientServiceCD
    ,t1.PatientServiceDSC
    ,t1.OriginalEventSubtypeDTS
    ,t1.OriginalEffectiveDTS
FROM
    Epic.Encounter.ADT_MGH t1
    join Epic.Encounter.ADT_MGH t2 on t1.LastInADTEventID=t2.EventID
    left join Epic.Patient.Identity_MGH t3 on (t1.PatientID=t3.PatientID  and t3.IdentityTypeID=67 )
    left join Epic.Encounter.PatientEncounter_MGH t4 on t4.PatientEncounterID=t1.PatientEncounterID
    left join Epic.Reference.Room t5 on t1.RoomID=t5.RoomID
    left join Epic.Reference.Bed t6 on t1.BedID=t6.BedID
WHERE
    t3.PatientIdentityID in ({})
