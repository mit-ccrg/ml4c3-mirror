select distinct
    t1.PatientEncounterID
    ,t1.HospitalAdmitDTS
    ,t1.HospitalDischargeDTS
    ,t1.HospitalAdmitTypeCD
    ,t1.HospitalAdmitTypeDSC
    ,t2.AdmitDiagnosisTXT
    ,t1.TemperatureFahrenheitNBR
    ,t1.HeartRateNBR
    ,t1.BloodPressureSystolicNBR
    ,t1.BloodPressureDiastolicNBR
    ,t1.RespirationRateNBR
    ,t1.WeightPoundNBR
    ,t1.HeightTXT
    ,t1.BodyMassIndexNBR
FROM
    Epic.Encounter.PatientEncounter_MGH t1
    left join Epic.Encounter.HospitalAdmitDiagnosis_MGH t2 on (t1.PatientEncounterID=t2.PatientEncounterID and t2.AdmitDiagnosisTXT is not null)

WHERE
    t1.PatientEncounterID in ({})
