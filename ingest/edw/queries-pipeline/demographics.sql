SELECT
    t2.PatientIdentityID as MRN
    ,t1.PatientID
    ,t4.PatientEncounterID
    ,t3.PatientRaceCD
    ,t3.PatientRaceDSC
    ,t1.BirthDTS
    ,t1.DeathDTS
    ,t1.EthnicGroupCD
    ,t1.EthnicGroupDSC
    ,t1.MaritalStatusCD
    ,t1.MaritalStatusDSC
    ,t1.PatientStatusCD
    ,t1.PatientStatusDSC
    ,t1.SexCD
    ,t1.SexDSC
FROM
    Epic.Patient.Patient_MGH t1
    left join Epic.Patient.Identity_MGH t2 on t1.PatientID=t2.PatientID
    left join Epic.Patient.Race_MGH t3 on t1.PatientID=t3.PatientID
    left join Epic.Encounter.PatientEncounter_MGH t4 on t1.PatientID=t4.PatientID
WHERE
    t4.PatientEncounterID in ({}) and t2.IdentityTypeID=67
