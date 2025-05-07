import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def butterworth_cauer_analog(n, f1, f2=None, R0=50,
                             filter_type='bandstop',
                             unit_C='uF', unit_L='mH',
                             first_element=None):
    """
    Compute the analog Cauer LC ladder for a Butterworth filter.

    Parameters:
    ----------
    n : int
        Filter order (total reactive elements = 2 * n).
    f1 : float
        Lower cutoff frequency in Hz (for lowpass/highpass, the single corner).
    f2 : float, optional
        Upper cutoff frequency in Hz (required for bandpass/bandstop).
    R0 : float
        System (image) impedance in Ohms.
    filter_type : {'lowpass','highpass','bandpass','bandstop'}
        Type of filter to synthesize.
    unit_C : {'F','uF'}
        Unit for output capacitances (Farad or microfarad).
    unit_L : {'H','mH'}
        Unit for output inductances (Henry or millihenry).
    first_element : {'shunt','series'}, optional
        Orientation of the first ladder element. Defaults to 'shunt'.

    Returns:
    --------
    analog : list of dict
        Each dict contains:
          - 'index': element number (1..2n)
          - 'orientation': 'shunt' or 'series'
          - 'C(unit_C)': capacitance value in chosen units
          - 'L(unit_L)': inductance value in chosen units
    """
    # unit conversion factors
    C_conv = {'F': 1, 'uF': 1e6}
    L_conv = {'H':  1, 'mH': 1e3}
    if unit_C not in C_conv or unit_L not in L_conv:
        raise ValueError("Unsupported unit_C or unit_L")
    if filter_type not in ('lowpass','highpass','bandpass','bandstop'):
        raise ValueError("Unsupported filter_type")

    # determine first element orientation (default shunt-first)
    fe = first_element if first_element in ('shunt','series') else 'shunt'

    # compute angular frequencies
    if filter_type in ('lowpass','highpass'):
        if f1 is None:
            raise ValueError("f1 (cutoff) required for lowpass/highpass")
        w_c = 2 * np.pi * f1
    else:
        if f1 is None or f2 is None:
            raise ValueError("f1 and f2 required for bandpass/bandstop")
        w1 = 2 * np.pi * f1
        w2 = 2 * np.pi * f2
        w0 = np.sqrt(w1 * w2)
        BW = w2 - w1

    # prototype g-values for Butterworth
    k = np.arange(1, n+1)
    g = 2 * np.sin((2*k - 1) * np.pi / (2*n))

    # build orientation sequence: shunt-first or series-first
    orient = []
    for i in range(n):
        if fe == 'shunt':
            orient.append('shunt' if i % 2 == 0 else 'series')
        else:
            orient.append('series' if i % 2 == 0 else 'shunt')

    analog = []
    for idx, (gk, ori) in enumerate(zip(g, orient), start=1):
        # initialize
        C_val = L_val = 0.0
        if filter_type == 'lowpass':
            # series L, then shunt C
            if ori == 'series':
                L_val = gk * R0 / w_c
            else:
                C_val = gk / (R0 * w_c)

        elif filter_type == 'highpass':
            # series C, then shunt L
            if ori == 'series':
                C_val = 1 / (gk * R0 * w_c)
            else:
                L_val = R0 / (gk * w_c)

        elif filter_type == 'bandpass':
            # shunt C, then series L
            if ori == 'shunt':
                C_val = gk / (R0 * BW)
                L_val = (R0 * BW) / (gk * w0**2)
            else:
                L_val = R0 * gk / BW
                C_val = BW / (R0 * gk * w0**2)

        else:  # bandstop
            # JS logic: shunt parallel LC, series series LC
            if ori == 'shunt':
                # parallel branch to ground
                C_val = (BW * 1e3 * gk) / (R0 * w0**2) * 1e-6
                L_val = (R0 / (gk * BW) / 1e3) * 1e-3
            else:
                # series branch in line
                C_val = (1e3 / (gk * R0 * BW)) * 1e-6
                L_val = ((BW * R0 * gk) / (w0**2) / 1e3) * 1e-3

        analog.append({
            'index': idx,
            'orientation': ori,
            f"C({unit_C})": C_val * C_conv[unit_C],
            f"L({unit_L})": L_val * L_conv[unit_L]
        })
    return analog


def print_analog_ladder(n, f1, f2=None, R0=50,
                        filter_type='bandstop',
                        unit_C='uF', unit_L='mH',
                        first_element=None):
    """
    Print the raw 2N-element LC ladder values.

    Parameters:
    ----------
    (same args as butterworth_cauer_analog)
    """
    analog = butterworth_cauer_analog(
        n, f1, f2, R0,
        filter_type, unit_C, unit_L,
        first_element
    )
    print(f"{filter_type.capitalize()} LC ladder:")
    header = f" #   Orient    C({unit_C})    L({unit_L})"
    print(header)
    print('-'*len(header))
    for e in analog:
        idx, ori = e['index'], e['orientation']
        C = e[f"C({unit_C})"]
        L = e[f"L({unit_L})"]
        print(f"{idx:2d}   {ori:6s}   {C:12.6f}   {L:10.6f}")


def design_digital_filter(order, f1, f2=None, fs=1.0, filter_type='bandstop'):
    """
    Design a digital Butterworth IIR filter.

    Parameters:
    ----------
    order : int       Order of IIR filter.
    f1 : float        Low cutoff (Hz) or single cutoff.
    f2 : float, opt.  High cutoff (Hz) for bandpass/bandstop.
    fs : float        Sampling frequency (Hz).
    filter_type : str Type of filter ('lowpass', etc.).

    Returns:
    --------
    b, a : ndarray   IIR numerator and denominator.
    eq : str         Difference equation string.
    """
    nyq = fs / 2.0
    if filter_type in ('lowpass','highpass'):
        Wn = f1 / nyq
    else:
        Wn = [f1/nyq, f2/nyq]
    b, a = butter(order, Wn, btype=filter_type, analog=False)
    b /= a[0]; a /= a[0]
    num = ' + '.join(f"{b[i]:.6g}*x[n-{i}]" if i>0 else f"{b[0]:.6g}*x[n]"
                    for i in range(len(b)))
    den = ' + '.join(f"{a[j]:.6g}*y[n-{j}]" for j in range(1,len(a)))
    eq = f"y[n] = ({num}) - ({den})"
    return b, a, eq


def apply_filter_and_plot(data, fs, f1, f2=None, order=4,
                          filter_type='bandstop'):
    """
    Apply digital filter and plot results.

    Parameters:
    ----------
    data : array-like   Input signal samples.
    fs : float         Sampling rate in Hz.
    f1, f2 : float     Cutoff frequencies.
    order : int        IIR filter order.
    filter_type : str  Filter type.

    Returns:
    --------
    y : ndarray        Filtered output signal.
    """
    b, a, eq = design_digital_filter(order, f1, f2, fs, filter_type)
    print("Digital difference equation:")
    print(eq)
    y = filtfilt(b, a, data)
    plt.figure(); plt.plot(data)
    plt.title('Raw Data'); plt.xlabel('n'); plt.ylabel('Amplitude'); plt.show()
    plt.figure(); plt.plot(y)
    plt.title(f'Filtered Data ({filter_type})'); plt.xlabel('n'); plt.ylabel('Amplitude'); plt.show()
    return y



def apply_filter_and_plot(data, fs, f1, f2=None, order=4,
                          filter_type='bandstop'):
    """
    Apply digital filter using scipy and plot raw vs filtered.
    """
    b, a, eq = design_digital_filter(order, f1, f2, fs, filter_type)
    print("Digital difference equation:")
    print(eq)
    y = filtfilt(b, a, data)
    plt.figure()
    plt.plot(data)
    plt.title('Raw Data')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.show()

    plt.figure()
    plt.plot(y)
    plt.title(f'Filtered Data ({filter_type})')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.show()
    return y


if __name__ == '__main__':
    
    # Global style tweaks
    plt.rcParams.update({
        'grid.linewidth': 2,
        # Remove the axes border by setting its width to 0
        'axes.linewidth': 0,
        'axes.labelpad': 12,
        'axes.titlepad': 12,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Tahoma', 'DejaVu Sans'],
        'font.size': 18, # base font size
        'font.weight': '500',
        'xtick.major.width': 0,
        'ytick.major.width': 0,
        'xtick.major.size': 0,
        'ytick.major.size': 0,
    })

    # ——— Load sensor data from CSV ———
    sensor_data = {}
    with open("example_data.csv", "r") as file:
        for line in file:
            parts = line.strip().split(',')
            if not parts or parts[0] == '':
                continue
            # Drop the "MEDIUM" prefix if present, our example has this tag infront of each sample
            if parts[0].upper() == "MEDIUM":
                parts = parts[1:]
            # Parse sensor–value pairs
            for i in range(0, len(parts), 2):
                sensor = int(parts[i])
                value = float(parts[i + 1])
                sensor_data.setdefault(sensor, []).append(value)

    # Temperatures for legend labels
    sensor_temps = [320, 100, 100, 100, 200, 200, 200, 320]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))
    b, a, eq = design_digital_filter(order=4, f1=150, fs=1200, filter_type='lowpass')
    for sensor in [0, 4, 1, 5, 2, 6, 3, 7]:
        samples = list(range(len(sensor_data[sensor])))
        y = filtfilt(b, a, sensor_data[sensor])
        ax.plot(
            samples,
            y,
            label=f"Sensor {sensor+1}: {sensor_temps[sensor]}°C",
            linewidth=4,
            solid_capstyle='round',
            solid_joinstyle='round'
        )

    # Remove the border/spines completely
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(True)

    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Sensor Reading")
    ax.set_title("Change in Sensor Readings Over Consecutive Samples")

    ax.set_xlim(left=0)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.16))

    plt.tight_layout()

    fig.savefig(
        "sensor_change_over_consecutive_samples.pdf",
        dpi=600,
        #bbox_inches='tight',
    )

    plt.show()
    
    # b, a, eq = design_digital_filter(order=4, f1=100, fs=1000, filter_type='lowpass')
    # print("Digital difference equation:")
    # print(eq)
    # y = filtfilt(b, a, data)
    # plt.figure()
    # plt.plot(data)
    # plt.title('Raw Data')
    # plt.xlabel('n')
    # plt.ylabel('Amplitude')
    # plt.show()

    # plt.figure()
    # plt.plot(y)
    # plt.title(f'Filtered Data ({filter_type})')
    # plt.xlabel('n')
    # plt.ylabel('Amplitude')
    # plt.show()

    

    # print_analog_ladder(n=6, f1=500, filter_type='lowpass',  R0=50)
    # print()
    # print_analog_ladder(n=6, f1=6000, filter_type='highpass', R0=50)
    # print()
    # print_analog_ladder(n=6, f1=2000, f2=6000, filter_type='bandpass',  R0=50)
    # print()
    # print_analog_ladder(n=6, f1=2000, f2=6000, filter_type='bandstop', R0=50)

    # digital demo: noisy sine
    # fs = 1000
    # t = np.arange(0,1,1/fs)
    # data = np.sin(2*np.pi*50*t) + 0.5*np.random.randn(len(t))
    # _ = apply_filter_and_plot(data, fs, f1=100, filter_type='lowpass', order=4)
    # _ = apply_filter_and_plot(data, fs, f1=25, filter_type='highpass', order=4)
    # _ = apply_filter_and_plot(data, fs, f1=45, f2=55, filter_type='bandpass', order=4)
    # _ = apply_filter_and_plot(data, fs, f1=40, f2=60, filter_type='bandstop', order=4)
