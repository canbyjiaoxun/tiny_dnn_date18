def float_to_bin(x):
  if x == 0:
    return "0" * 32
  w, sign = (float.hex(x), 0) if x > 0 else (float.hex(x)[1:], 1)
  mantissa, exp = int(w[4:17], 16), int(w[18:])
  #return "{0}{1:008b}{2:023b}".format(sign, exp + 127, mantissa)
  return "{1:008b}".format(exp+127)


print float_to_bin(0.77) 
