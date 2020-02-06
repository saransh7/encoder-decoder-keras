if __name__ == "__main__":
    import os
    from datetime import datetime
    import time
    import support.config as c
    from utils import train_util

    print('Code Started at ' + datetime.fromtimestamp(time.time()
                                                      ).strftime('%Y-%m-%d %H:%M:%S'))

    os.chdir(c.project_dir)
    train_util.train_model()

    print('Code Ended  at ' + datetime.fromtimestamp(time.time()
                                                     ).strftime('%Y-%m-%d %H:%M:%S'))
