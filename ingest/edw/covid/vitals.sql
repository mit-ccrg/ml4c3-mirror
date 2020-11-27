select distinct 'null',
    t1.PatientEncounterID
    ,[TemperatureFahrenheitNBR]
    ,[HeartRateNBR]
    ,[BloodPressureSystolicNBR]
    ,[BloodPressureDiastolicNBR]
    ,[RespirationRateNBR]
    ,[WeightPoundNBR]
    ,[HeightTXT]
    ,[BodyMassIndexNBR]
    ,OxygenSaturationNBR
from [Epic].[Encounter].[PatientEncounter_MGH] t1
left outer join [Epic].[Encounter].[PatientEncounter2_MGH] t2 on t1.PatientEncounterID=t2.PatientEncounterID --spo2 is contained in the PatientEncounter2 table, whereas the other fields are in the PatientEncounter table
WHERE t1.PatientEncounterID in ()
