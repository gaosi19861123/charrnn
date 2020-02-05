class config(object):

    data_path = "data/"
    pickle_path = "tang.npz"
    author = None
    constrain = None
    category = "poet.tang"

    lr = 1e-3
    user_gpu = True
    epoch = 20
    batch_size = 128 
    maxlen = 125
    plot_every =20
    use_env = True
    env = "poetry"
    max_gen_len = 200
    debug_file = "/tmp/debugp"
    model_path = None