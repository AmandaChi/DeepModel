import tensorflow as tf
import collections
import re
def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  #print(init_vars)
  flag = True
  for x in init_vars:
    (name, var) = (x[0], x[1])
   # print(name,var)
    name_ = name.replace("device_0/","")
    if name in name_to_variable and var == name_to_variable[name].shape:
      #continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1
    elif "device_0/" + name in name_to_variable and var == name_to_variable["device_0/"+name].shape:
        assignment_map[name] = "device_0/" + name
        initialized_variable_names["device_0/" + name] = 1
        initialized_variable_names["device_0/" + name + ":0"] = 1
    elif name_ in name_to_variable and var == name_to_variable[name_].shape:
        assignment_map[name] = name_
        initialized_variable_names[name_] = 1
        initialized_variable_names[name_ + ":0"] = 1
    else:
        flag = False
  if not flag:
    print("[WARNING] not all parameter loaded! But it's OK if expected")
  return (assignment_map, initialized_variable_names)


