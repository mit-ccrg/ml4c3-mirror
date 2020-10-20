from .summary_writer import PreTensorizeSummaryWriter


def pre_tensorize_summary(args):
    summary = PreTensorizeSummaryWriter(
        args.path_bedmaster, args.path_edw, args.path_xref,
    )

    summary.write_pre_tensorize_summary(
        args.output_dir,
        args.summary_stats_base_name,
        args.signals,
        args.detailed_bm,
        args.no_xref,
    )
