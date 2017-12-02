def in_range(val, lower, upper):
    return lower <= val <= upper


class Deadzone:
    def __init__(self, critical_value=20, pot_range=1023):
        self.critical_value = critical_value
        self.pot_range = pot_range
        self.active = False
        self.in_critical_zone = False

    def activate(self):
        self.active = True

    def clean_val(self, val):
        if self.active:
            if not self.in_critical_zone and not in_range(val, self.critical_value,
                                                          self.pot_range - self.critical_value):
                self.in_critical_zone = True

            if self.in_critical_zone and (
                        in_range(val, self.critical_value, self.critical_value * 3) or in_range(val,
                                                                                                self.pot_range - self.critical_value * 3,
                                                                                                self.pot_range - self.critical_value)):
                self.in_critical_zone = False

            if self.in_critical_zone and in_range(val, self.critical_value * 3,
                                                  self.pot_range - self.critical_value * 3):
                val = 0

            return val
        else:
            return 0
