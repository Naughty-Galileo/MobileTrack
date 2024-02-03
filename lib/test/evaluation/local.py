from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/root/MobileTrack/data/got10k_lmdb'
    settings.got10k_path = '/root/autodl-tmp'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/root/MobileTrack/data/lasot_lmdb'
    settings.lasot_path = '/root/MobileTrack/data/lasot'
    settings.network_path = '/root/MobileTrack/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/root/MobileTrack/data/nfs'
    settings.otb_path = '/root/datasets/OTB100'
    settings.prj_dir = '/root/MobileTrack'
    settings.result_plot_path = '/root/MobileTrack/test/result_plots'
    settings.results_path = '/root/MobileTrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/root/MobileTrack'
    settings.segmentation_path = '/root/MobileTrack/test/segmentation_results'
    settings.tc128_path = '/root/MobileTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/root/MobileTrack/data/trackingNet'
    settings.uav_path = '/root/MobileTrack/data/UAV123'
    settings.vot_path = '/root/MobileTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

