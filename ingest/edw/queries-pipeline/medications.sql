SELECT
    t2.PatientIdentityID as MRN
    ,t1.PatientEncounterID as CSN
    ,t1.OrderStatusDSC
    ,t1.MedicationDSC
    ,t1.DoseUnitDSC
    -- ,t1.MinimumDoseAMT
    -- ,t1.MaximumDoseAMT
    ,t1.DiscreteDoseAMT
    -- ,t1.DiscreteFrequencyDSC
    ,t1.PatientWeightBasedDoseFLG
    ,t3.MedicationTakenDTS
    ,t3.MARActionDSC
    ,t3.RouteDSC
    -- ,t3.[SiteDSC]
    ,t3.[InfusionRateNBR]
    ,t3.[InfusionRateUnitDSC]
    ,t3.[DurationNBR]
    ,t3.[DurationUnitDSC]
    ,t4.TherapeuticClassDSC
    ,t4.PharmaceuticalClassDSC
    ,t4.SimpleGenericDSC
FROM
    [Epic].[Orders].[Medication_MGH] t1
    inner join Epic.Patient.Identity_MGH t2 on t1.PatientID = t2.PatientID
    left join Epic.Clinical.AdministeredMedication_MGH t3 on (t1.OrderID=t3.OrderID)
    left join Epic.Reference.Medication t4 on (t1.MedicationID = t4.MedicationID)
WHERE
    t1.PatientEncounterID in ({}) and t2.IdentityTypeID = 67
    --and MedicationTakenDTS is not null
    --and t3.MARActionDSC in ('Given', 'New Bag', 'Rate Change', 'Stopped','Restarted','Same Bag', 'Rate Verify')
