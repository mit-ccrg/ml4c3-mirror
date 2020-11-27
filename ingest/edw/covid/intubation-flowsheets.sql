SELECT pid.PatientIdentityID as MRN, t5.PatientID,InpatientDataID,t4.FlowsheetDataID,t4.FlowsheetMeasureID
			,FlowsheetMeasureNM,DisplayNM,RecordedDTS,MeasureTXT
FROM [Epic].[Clinical].[FlowsheetMeasure_MGH] t4
  INNER join Epic.Clinical.FlowsheetRecordLink_MGH t5 on t4.FlowsheetDataID=t5.FlowsheetDataID
  INNER join Epic.Clinical.FlowsheetGroup_MGH t6 on t4.FlowsheetMeasureID=t6.FlowsheetMeasureID
	and t6.FlowsheetMeasureID in (
	--- PEEP AND FiO2 ---
	'10850','1120010102', '316050', '33306','1120100038',
	'1120100035','15706','301550','301620','3040100913','3040100914','3040101005','3040102560',
	'3040102579','3040102591','3040102592','3041332601','304140011',
	'304140036','304140047','304140052','3043300800','3043300801','304490490','304561008','3047000005',
	'316100','316320','7074386','7074387',
	--- OXYGEN DEVICE ---
	'301030'
	)
  Inner join [Epic].[Patient].[Identity_MGH] pid on pid.PatientID = t5.PatientID and [IdentityTypeID] = 67
  ---where DATEDIFF(day,RecordedDTS, GETDATE()) = 1
  ---where t5.PatientID in ({})
