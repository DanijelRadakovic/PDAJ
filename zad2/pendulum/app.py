from celery import Celery
from celery.signals import worker_ready


DEFAULT_RESOLUTION = 6
DEFAULT_TMAX = 30
DEFAULT_DT = 0.01

DEFAULT_L1 = 1
DEFAULT_L2 = 1
DEFAULT_M1 = 1
DEFAULT_M2 = 1

app = Celery('pendulum')
app.config_from_object('pendulum.celeryconfig')


if app.conf.AM_I_SERVER:
    @worker_ready.connect
    def bootstrap(**kwargs):
        from .tasks.server import seed_computations

        delay_time = 10 # seconds
        print "Getting ready to automatically seed computations in %d seconds..." % delay_time
        seed_computations.apply_async(countdown=delay_time)


if __name__ == '__main__':
    app.start()
