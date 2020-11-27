--- ADT query ---
    SELECT [EventID]
      ,[ADTEventTypeDSC]
      ,[ADTEventSubtypeDSC]
      ,[DepartmentID]
      ,[DepartmentDSC]
      ,[RoomID]
      ,[BedID]
      ,[EffectiveDTS]
      ,ADT.[PatientID]
      ,PID.PatientIdentityID as MRN
      ,ADT.[PatientEncounterDateRealNBR]
      ,ADT.[PatientEncounterID]
      ,[PatientClassCD]
      ,[PatientClassDSC]
      ,[PatientServiceCD]
      ,[PatientServiceDSC]
      ,ADT.[AccommodationCD]
      ,Acc.AccommodationDSC
      ,[AccommodationReasonDSC]
      ,[PatientSummaryTypeCD]
      ,[PatientSummaryTypeDSC]
      ,[FromBaseClassCD]
      ,[FromBaseClassDSC]
      ,[ToBaseClassCD]
      ,[ToBaseClassDSC]
      ,iso.IsolationDSC as IsolationStatus
      ,iso.IsolationAddedDTS as IsolationStartDTS
      ,iso.IsolationRemovedDTS as IsolationEndDTS
  FROM [Epic].[Encounter].[ADT_MGH] ADT
  left join (select distinct AccommodationCD, AccommodationDSC
                From Epic.Encounter.PatientEncounterHospital_MGH) Acc On ADT.AccommodationCD = Acc.AccommodationCD
    join [Epic].[Patient].[Identity_MGH] PID ON PID.PatientID = ADT.PatientID and PID.IdentityTypeID = 67
    Left JOIN [Epic].[Encounter].[HospitalIsolation_MGH] iso
            on iso.PatientEncounterID = ADT.PatientEncounterID
  Where ADT.PatientID in ({})
  order by PatientID, ADT.PatientEncounterID, ADT.EffectiveDTS
