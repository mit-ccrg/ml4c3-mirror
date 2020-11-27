--- Generic ICD diagnostic data query---
SELECT
[PatientEncounterID]
,[PatientID]
,ContactDTS
,t1.[DiagnosisID]
,t2.[DiagnosisNM]
,t2.[DiagnosisGroupDSC]
,[CommentTXT]
,[ICD9DiagnosisStageID]
,[CurrentICD9ListTXT]
,[CurrentICD10ListTXT]
,[ExternalID]
,[EDDiagnosisID]
FROM [Epic].[Encounter].[PatientEncounterDiagnosis_MGH] t1
inner join [Epic].[Reference].[ICDDiagnosis] t2 ON t1.DiagnosisID = t2.DiagnosisID
--- conditions ---
WHERE PatientID IN ({})
