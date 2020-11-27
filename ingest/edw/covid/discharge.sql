SELECT distinct 'null',t2.PatientIdentityID, t1.PatientEncounterID
      ,t1.HospitalDischargeDTS
from [Encounter].[PatientEncounter_MGH]  t1
inner join epic.Patient.Identity_MGH t2 on  (t1.PatientID=t2.PatientID  and t2.IdentityTypeID in (227,67))
where t1.PatientEncounterID in () -- include all (old and new) encounters
