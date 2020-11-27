SELECT distinct 'null', t2.PatientIdentityID as MRN
      ,t1.[DeathDTS]
FROM [Epic].[Patient].[Patient_MGH] t1
inner join epic.Patient.Identity_MGH t2 on  (t1.PatientID=t2.PatientID  and t2.IdentityTypeID in (227,67))
where t2.PatientIdentityID in () -- include ALL MRNs (old and new)
