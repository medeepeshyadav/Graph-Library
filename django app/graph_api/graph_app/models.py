from django.db import models
# import random
# import hashlib
import uuid

# class GUIDModel(models.Model):
#     guid = models.CharField(primary_key=True, max_length=50, blank=True)

#     def save(self, *args, **kwargs):
#         if not self.guid:
#             self.guid = hashlib.sha1(str(random.random())).hexdigest()

#         super(GUIDModel, self).save(*args, **kwargs)

class Order(models.Model):

    file_id = models.UUIDField(primary_key=True, default=uuid.uuid4() , editable=False, blank=True)
    in_file_loc = models.CharField(max_length=300, default=None)
    out_file_loc = models.CharField(max_length=300, default=None, null=True, blank=True)
    status = models.CharField(max_length=50, default='ready')
    graph_type = models.CharField(max_length=20, default='directed')
    feature_type = models.CharField(max_length=20, default='basic')
    test_size = models.DecimalField(max_digits=2, decimal_places=2, default=0.2)
    n_jobs = models.IntegerField(default=1)

    def __str__(self):
        return str(self.file_id)
    
    def get_absolute_url(self):
        return "uploaded/%i/" % self.file_id

class Parameter(models.Model):

    input_file = models.CharField(max_length=300)
    graph_type = models.CharField(max_length=20)
    feature_type = models.CharField(max_length=20)
    test_size = models.DecimalField(max_digits=2, decimal_places=2)
    n_jobs = models.IntegerField()

    def __str__(self):
        return self.input_file.split("/")[-1]

