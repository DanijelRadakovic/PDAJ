from celery import Celery
from celery.signals import worker_ready


app = Celery('export_beam_integrals')
app.config_from_object('export_beam_integrals.celeryconfig')


if app.conf.AM_I_SERVER:
    @worker_ready.connect
    def bootstrap(**kwargs):
        from .tasks.server import seed_computations

        delay_time = 10 # seconds
        print "Getting ready to automatically seed computations in %d seconds..." % delay_time
        seed_computations.apply_async(countdown=delay_time)


if __name__ == '__main__':
    app.start()
