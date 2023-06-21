from utils.wutils import *    # infinity way
# from wutils_ori import *       # VQGAN
# from wutils_tmp import *   # Global Optimize


class BasicArgs:
    # default settings
    resume = True
    use_tqdm = True
    debug = False

    # Chenfei's personal Nodes
    if platform.node() == 'Moymix-PC':
        logger.info("检测到实验室台式机%s" % platform.node())
        root_dir = '/root/data/G'
        debug = True

    elif platform.node() == 'chenfei':
        logger.info("检测到笔记本%s" % platform.node())
        root_dir = 'E:/data/G'
        debug = True

    elif platform.node() == 'MININT-UG6RO8P':
        logger.info("Detected MSRA PC Node %s." % platform.node())
        root_dir = 'D:/data/G'
        debug = True

    # PAI Platform Nodes
    elif platform.node().startswith('gcr-openpai'):
        logger.info("Detected PAI V100 Node %s. This is a one-node for debugging." % platform.node())
        root_dir = '/f_ndata/G/'

    elif platform.node().startswith('a100compute'):
        logger.info("Detected PAI A100 Node %s." % platform.node())
        root_dir = '/f_ndata/G/'

    # ITP Platform Nodes
    elif platform.node().endswith('master-0') or \
            platform.node().startswith('phlrr') or \
            platform.node().startswith('PHLRR'):
        logger.info("Detected ITP Common Node %s." % platform.node())
        root_dir = '/f_ndata/G/'
        os.environ['MASTER_ADDR'] = os.environ['MASTER_IP']
        os.environ['MASTER_PORT'] = '12306'

    elif platform.node().startswith('az-eus-v100-16gb') or platform.node().startswith('az-sea-v100-16gb'):
        logger.info("Detected ITP Common Node %s." % platform.node())
        root_dir = '/f_ndata/G/'
        os.environ['MASTER_ADDR'] = os.environ['MASTER_IP']
        os.environ['MASTER_PORT'] = '12306'

    elif platform.node().startswith('msrit'):
        logger.info("Detected ITP Hyper A100 Node %s." % platform.node())
        root_dir = '/f_ndata/G/'

    elif platform.node().startswith('GCRHYP'):
        logger.info("Detected ITP Hyper V100 Node %s." % platform.node())
        root_dir = '/f_ndata/G/'

    # Singularity Nodes / A100
    elif platform.node().startswith('node'):
        logger.info("Detected Singularity Node %s." % platform.node())
        root_dir = '/scratch/f_ndata/G/'

    # System Super Bench Nodes
    elif platform.node().startswith('sb-nlp'):
        logger.info("Detected System SuperBench Node %s." % platform.node())
        root_dir = '/f_ndata/G/'

    else:
        logger.info("Detected unknown Node %s." % platform.node())
        root_dir = '/f_ndata/G/'


    @staticmethod
    def parse_config_name(config_filename):
        """
        Example:
            Args:
                config_filename: 'config/t2i/t2i4ccF8S256.py'
            Return:
                task_name: 't2i'
                model_name: 't2i4ccF8S256'
        """
        task_name, filename = os.path.normpath(config_filename).split(os.path.sep)[-2:]
        model_name = filename.split('.')[0]
        return task_name, model_name

def init_model_from_config(cfg_path='config/vae_kl/VAE_KL_F8D4.py'):
    cfg = import_filename(cfg_path)
    Model_net, Args_net = cfg.Net, cfg.args
    model = Model_net(Args_net)

    pretrained_model = getattr(Args_net, 'pretrained_path', None)
    if pretrained_model:
        print(f"load pretrained model: {pretrained_model} for config {cfg_path}...")
        model.load_state_dict(file2data(pretrained_model, map_location='cpu'), strict=False)

    return model