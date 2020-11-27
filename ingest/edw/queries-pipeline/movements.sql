SELECT distinct
    t3.PatientIdentityID as MRN
    ,t1.PatientEncounterID
    ,t4.HospitalAdmitDTS
    ,t4.HospitalDischargeDTS
    ,t2.EffectiveDTS as TransferInDTS
    ,t1.EffectiveDTS as TransferOutDTS
    ,t1.ADTEventTypeDSC
    ,t1.DepartmentID
    ,t1.DepartmentDSC
    ,t1.RoomID
    ,t1.BedID
    ,t5.BedLabelNM

FROM
    Epic.Encounter.ADT_MGH t1
    join Epic.Encounter.ADT_MGH t2 on t1.LastInADTEventID=t2.EventID
    left join Epic.Patient.Identity_MGH t3 on t1.PatientID=t3.PatientID
    left join Epic.Encounter.PatientEncounter_MGH t4 on t4.PatientEncounterID=t1.PatientEncounterID
    left join Epic.Reference.Bed t5 on t1.BedID=t5.BedID
WHERE
    t1.PatientEncounterID in ({}) and t3.IdentityTypeID=67
