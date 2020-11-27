SELECT distinct 'null',t1.PatientEncounterID, t1.MedicationID, t1.MedicationDSC, t1.MedicationDisplayNM
FROM [Epic].[Orders].[Medication_MGH] t1
INNER JOIN (VALUES ()) --value pairs should be of form (encid,admitdate)
  AS pairs (encid, admitdate)
  ON (CAST(pairs.admitdate as DATE)=CAST(t1.OrderInstantDTS as DATE))
  and encid=t1.PatientEncounterID

/***
Note: MedicationDisplayNM is sometimes NULL, but other times it contains useful information that MedicationDSC is missing, so I included both fields
If you want to only include meds that were actually administered, inner join to [Epic].[Clinical].['AdministeredMedication_MGH']
filter by MARActionDSC  in (Give, Restarted,New Bag, Same Bag, Rate Change, Rate Verify)
***/
