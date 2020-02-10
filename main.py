if __name__ == "__main__":
    import os
    import time
    import argparse
    import support.config as c
    from utils import train_util
    from datetime import datetime
    from utils import prediction_util

    os.chdir(c.project_dir)
    print('Code Started at ' + datetime.fromtimestamp(time.time()
                                                      ).strftime('%Y-%m-%d %H:%M:%S'))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--reply', help='reply to my message', action='store_true')
    parser.add_argument('-m', '--message', help='message', type=str)
    args = parser.parse_args()

    if args.reply:
        if args.message:
            print(args.message)
            prediction_util.reply_text(args.message)
        else:
            print('replying to sample input')
            prediction_util.reply_text(c.sample_input)
    else:
        print('starting training...')
        train_util.train_model()

    print('Code Ended  at ' + datetime.fromtimestamp(time.time()
                                                     ).strftime('%Y-%m-%d %H:%M:%S'))
