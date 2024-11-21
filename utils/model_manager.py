import torch
from datetime import datetime



class ModelManager:
  def __init__(self):
    pass
  def save_to_disk(self,model,name=None,path="./models"):
    if name is None :
      name=self._get_cur_timestamp_str()
    torch.save(model.state_dict(),'./model/'+name+'.pt')
    return name

  def _get_cur_timestamp_str(self):
    return datetime.now().strftime('%Y-%m-%d_%H_%M_%S')