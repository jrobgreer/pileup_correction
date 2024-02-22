import numpy as np
import ROOT
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.signal import argrelextrema
import getwave as gw
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import ROOT
import array
import gc
import pandas as pd
import time
from itertools import chain

plt.rcParams['figure.figsize'] = (20, 4)
# COMPASS cannot timestamp events
# Changed setup to manually timestamp

# For neutron generator setup, we have an external trigger based upon when the neutron pulse occurs
# The TTL pulse from the generator goes into a gate generator
# The gate produced is sent into TRG-IN on DT5730
# Then set up


class Pulse:

    def __init__(self, pulse_collection, pulse_timestamps, index):

        # Now we've expanded the record length, REMAINING PULSE AREA after subtraction is no longer
        # a good measure of fit quality for the final pulse, there will always be an undershoot
        self.record = np.zeros(2000)
        self.record[0:len(pulse_collection[index])] = pulse_collection[index]
        # Currently a crude timestamp based on the timestamp of the first trigger - TODO need to add timestamping for events within a record - TTT + the offset of the extra pulse
        self.timestamp = pulse_timestamps[index]
        self.event_id = index

        # Flag for throwing away bad pulses
        self.pulse_suitable = True

        # True timestamps will contain multiple timestamps from many pulses in a waveform
        self.true_timestamps = []
        self.areas = []
        self.chi2 = []
        self.ndf = []
        self.par0 = []
        self.par1 = []
        self.par2 = []
        self.par3 = []
        self.record_id = []
        self.pulse_count = []
        self.remaining_pulse_area = []
        self.time_to_fit = []
        self.fitresultstatus = []
        self.rise_point = []
        self.raw_integral = []

        # NOTE may be worth getting rid of these
        self.time = np.linspace(0, 1029, 1030)
        self.record_length = len(pulse_collection[index])

    def fft(self, plotting=False):
        '''Produce FFT spectrum - optional plotting'''
        self.yf = fft(self.record)
        # 4ns between samples, should probably make this more general
        self.xf = fftfreq(self.record_length, 4e-9)

        if plotting == True:
            plt.plot(self.xf, np.abs(self.yf))
            plt.show()
            plt.close()

    def butter_lowpass_filtfilt(self, cutoff, fs, order=5, plotting=False):
        '''Low pass filter'''

        def butter_lowpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, self.record)

        if plotting == True:
            # Plot pre and post filter waveforms
            # plt.plot(self.record, label='Original')
            self.record = y
            plt.plot(self.record, label=f'Filtered @ {cutoff} Hz')
            # plt.legend()
            # plt.show()
            # plt.close()

        self.record = y

    def plot(self):
        plt.plot(self.record, label='Original')
        # plt.show()

    def raw_int(self):
        integral = np.sum(self.record[self.rise_indices[0]:])
        # print("Raw integral : ", integral)
        self.rawint = integral
        self.raw_integral.append(integral)

    def get_peaks2(self, plotting=False, min_dist_between_peaks=15, gradient_threshold=25, moving_av_filt=8):
        '''Find the rise points based upon gradient, for use as fit limits in fit2
           min_dist_between_peaks - allows removal of peaks too closely spaced for fitting
           gradient_threshold - what gradient defines a rise
           moving_av_filt - spacing for moving avg filtering'''

        # Check for encroaching pulses from previous record
        if np.median(self.record[:10]) > 500:
            self.pulse_suitable = False
            print("Encroaching previous pulse, skip to next...")
            return

        # plt.close()
        # plt.plot(self.record, label='Original wfm')
        # plt.show()

        def moving_average(a, n=moving_av_filt):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        self.record = moving_average(self.record)

        self.gradient = np.gradient(self.record)

        # Remove strange noise shoulders on first pulse - need to find source of these, if noise, ok, but
        # what if these are real pulses tangled up in the rise of the bigger pulse?
        # To do this, set anything before the first peak to 0, for a sharp rise
        try:
            first_neg_grad = np.argmax(self.gradient < -20)
            print(first_neg_grad)
            threshold_idx = np.argmax(self.record[:first_neg_grad])
            print(threshold_idx)
            self.record[:threshold_idx] = 0
            # plt.plot(self.record, label='Shoulder removed')
            # plt.legend()
            # plt.show()
            # self.record[:first_neg_grad] = 0

        except ValueError:
            print("Gradient never less than -20 ")
            print("First pulse shoulder not corrected")
            # plt.plot(self.record, label='Edited')
            # plt.axvline(first_neg_grad)
            # plt.plot(self.gradient, label='gradient')
            # plt.legend()
            # plt.show()
            # self.pulse_suitable = False

        # plt.plot(self.record, label='Edited')
        # plt.axvline(first_neg_grad)
        # plt.plot(self.gradient, label='gradient')

        # Recalculate gradient before finding rises
        self.gradient = np.gradient(self.record)
        # Locate rise of the pulse from a spike in gradient
        rise_indices = np.asarray(
            (self.gradient > gradient_threshold)).nonzero()

        # Need to select rise from clusters detected around peak, use first
        prev_rise_index = None
        rise_indices_clustered = []
        # print(rise_indices)

        for i, rise_index in enumerate(np.array(rise_indices).flatten()):
            # print("Rise index:")
            # print(rise_index)
            if prev_rise_index == None:
                # print("Starting clustering")
                prev_rise_index = rise_index
                rise_indices_clustered.append(rise_index)

            # Clusters of points flagged as rises, only need one to fit from
            # Look through the rise indices chosen
            # If too close together, discard
            elif rise_index > prev_rise_index + min_dist_between_peaks:
                # print(rise_index, " compared with ", prev_rise_index)
                prev_rise_index = rise_index
                rise_indices_clustered.append(rise_index)

        # print(rise_indices_clustered)
        # plt.scatter(x=rise_indices, y=gradient[rise_indices], s=10, marker='*', color='r', label='Spikes')
        # plt.scatter(x=rise_indices_clustered, y=gradient[rise_indices_clustered], s=100, marker='+', color='green', label='Rises')
        # plt.scatter(x=rise_indices_clustered, y=self.record[rise_indices_clustered], s=100, marker='+', color='green', label='Rises')
        # plt.plot(self.record)
        # plt.legend()
        # plt.show()

        self.rise_indices = rise_indices_clustered
        self.rise_amplitudes = self.record[rise_indices_clustered]

        # plt.scatter(self.rise_indices, self.rise_amplitudes)
        # plt.legend()
        # plt.show()

    def get_peaks(self, plotting=False, interpolation_length=10000, minimum_distance=10, threshold=500, diff_scaling=10, minimum_time=0):
        '''Differentiate and locate peaks based on gradient, second differential, and minimum distance apart
           Disgusting code, needs tidying, but it works most of the time'''

        # Calculate gradient
        grad = []

        for i in range(len(self.record)-1):

            # Factor of diff_scaling used for visibility in plot
            # 8 ns per sample
            grad.append(diff_scaling*(self.record[i+1]-self.record[i])/8)

        grad = np.array(grad)

        # Calculate gradient of gradient
        gradgrad = []

        for i in range(len(grad)-1):
            gradgrad.append(diff_scaling*(grad[i+1]-grad[i])/8)

        gradgrad = np.array(gradgrad)

        # Interpolate the gradient, to get values closer to zero for steep changes - show maxima
        fine_grad = interp1d(np.linspace(0, 1029, 1029), grad)
        fine_gradgrad = interp1d(np.linspace(0, 1029, 1028), gradgrad)
        pulse_interp = interp1d(np.linspace(0, 1030, 1030), self.record)
        time_interp = interp1d(np.linspace(0, 1030, 1030), self.time)

        # Create arrays
        fine_grad = fine_grad(np.linspace(0, 1029, interpolation_length))
        fine_gradgrad = fine_gradgrad(
            np.linspace(0, 1029, interpolation_length))
        pulse_interp = pulse_interp(np.linspace(0, 1029, interpolation_length))
        time_interp = time_interp(np.linspace(0, 1029, interpolation_length))
        # print(time_interp)

        print("Making event mask")

        # Check the gradient is negative and small - ie we have reached a peak, and heading back down after
        # Also check if the pulse value is above threshold
        # Check second diff is negative, we are expecting gradient to get more negative
        # print(time_interp)
        # print(minimum_time)
        # print(time_interp[time_interp>minimum_time])

        # plt.close()
        # plt.axvline(x=minimum_time, label='Cutoff Time', linestyle='-')
        # plt.plot(self.record, label='Original')
        # plt.plot(grad, label='Gradient')
        # plt.plot(gradgrad, label='Double Gradient')
        # ##plt.scatter(peak_times, peak_heights, color='black', marker='*', s=100, label='Peaks')
        # plt.legend()
        # plt.show()
        # plt.close()

        event_mask = (fine_grad < 0) & (fine_grad > -500) & (pulse_interp >
                                                             threshold) & (fine_gradgrad > -100) & (time_interp > minimum_time)

        # plt.scatter(np.linspace(0,1029, interpolation_length)[event_mask], pulse_interp[event_mask], label='FULLMASK', marker='+', color='r', s=100)

        # Take first instance in found peaks to be THE peak, can then be subtracted from rest of them
        peak_times = []
        peak_heights = []

        # Take first peak as a peak
        # print(np.linspace(0,1029, interpolation_length)[event_mask])

        peak_t = np.linspace(0, 1029, interpolation_length)[event_mask][0]
        peak_v = pulse_interp[event_mask][0]

        peak_times.append(peak_t)
        peak_heights.append(peak_v)

        for peak_idx in range(len(np.linspace(0, 1029, interpolation_length)[event_mask])):

            if np.linspace(0, 1029, interpolation_length)[event_mask][peak_idx] > peak_t+minimum_distance:
                peak_t = np.linspace(0, 1029, interpolation_length)[
                    event_mask][peak_idx]
                peak_times.append(peak_t)

                peak_v = pulse_interp[event_mask][peak_idx]
                peak_heights.append(peak_v)

            else:
                continue

        if plotting == True:
            plt.axvline(x=minimum_time, label='Cutoff Time', linestyle='-')
            plt.plot(self.record, label='Original')
            plt.plot(grad, label='Gradient')
            plt.plot(gradgrad, label='Double Gradient')
            plt.scatter(peak_times, peak_heights, color='black',
                        marker='*', s=100, label='Peaks')
            plt.legend()
            plt.show()
            plt.close()

        self.peak_times = peak_times
        self.peak_heights = peak_heights

    def fit2(self, closest_distance=25, fit_options='QRS'):

        # plt.plot(self.record, label='Remove shoulder + mov avg')

        # plt.scatter(self.rise_indices,
        #             self.record[self.rise_indices], marker='+', s=100, label='Rise')
        # plt.legend()
        # plt.show()
        # plt.close()

        # For pileup waveforms, check distance between rises
        # If too small, separation tricky, skip over
        try:
            gaps_between_pulses = np.diff(self.rise_indices)
            print(gaps_between_pulses)
        except AttributeError:
            print("Bad pulse, do not fit, move to next...")

        try:
            print(np.min(gaps_between_pulses))
            print(closest_distance)
            if np.min(gaps_between_pulses) < closest_distance:
                print("Pulses too close, skipping to next pulse")
                self.pulse_suitable = False

        except ValueError:
            print("One pulse waveform, continuing with fit...")

        except UnboundLocalError:
            print("Diffs not calculated")

        if self.pulse_suitable == True:

            # plt.close()
            # # plt.scatter(x=self.rise_indices, y=self.gradient[self.rise_indices], s=10, marker='*', color='r', label='Spikes')
            # # plt.scatter(x=self.rise_indices, y=self.gradient[self.rise_indices], s=100, marker='+', color='green', label='Rises')
            # plt.scatter(x=self.rise_indices,
            #             y=self.record[self.rise_indices], s=100, marker='+', color='green', label='Rises')
            # plt.plot(self.record, label='Pulse')
            # plt.legend()
            # plt.show()

            def guo_fit(x, par):
                '''par[0] - A
                   par[1] - t0
                   par[2] - theta1
                   par[3] - theta2'''
                return par[0]*(np.exp(-(x[0]-par[1])/par[2]) - np.exp(-(x[0]-par[1])/par[3]))

            def perform_fit(rise_idx, fit_end, idx):
                '''fit_start and fit_end - indices corresponding to the pulse fit range
                   idx for reference to other stuff'''

                graph = ROOT.TGraph(len(self.record[rise_idx:]), np.linspace(
                    rise_idx, len(self.record), len(self.record)-rise_idx), self.record[rise_idx:])

                # graph.SetMarkerStyle(20)
                # graph.SetMarkerSize(1)
                # graph.SetMarkerColor(4)

                # print("AMPLITUDE: ", np.max(self.record[rise_idx:fit_end]))
                # Only fit up to next rise
                fit_function = ROOT.TF1(
                    'guo_fit', guo_fit, rise_idx, fit_end, 4)
                fit_function.SetParameters(
                    1.5*np.max(self.record[rise_idx:fit_end]), self.rise_indices[idx], 50, 10)

                fit_function.FixParameter(1, rise_idx)
                fit_function.SetParLimits(0, np.max(
                    1*self.record[rise_idx:fit_end]), 2*np.max(self.record[rise_idx:fit_end]))
                # fit_function.SetParLimits(1, rise_idx-2, rise_idx+2)
                fit_function.SetParLimits(2, 20, 300)
                fit_function.SetParLimits(3, 2, 20)

                start_time = time.time()

                fit_result = graph.Fit(fit_function, fit_options)

                self.fitresultstatus.append(fit_result.Status())

                end_time = time.time()
                fit_time = end_time - start_time

                # print("---------------------------------------------------------------------------")
                print("Time to fit: ", fit_time)
                print("Fit status: ", fit_result.Status())
                # print("---------------------------------------------------------------------------")

                # canvas = ROOT.TCanvas(
                #     "canvas", "Guo Model Pulse Fit", 1000, 600)
                # graph.Draw("AP")
                # fit_function.Draw("same")
                # canvas.Update()
                # canvas.Draw()

                # input("Enter to cont")

                fitted_pulse = np.array([guo_fit([t], [fit_function.GetParameter(i) for i in range(
                    fit_function.GetNpar())]) for t in np.linspace(rise_idx, len(self.record), len(self.record)-rise_idx)])

                print(len(self.record))

                print(np.linspace(rise_idx, len(self.record),
                      len(self.record)-rise_idx))
                print(
                    len(np.linspace(rise_idx, len(self.record), len(self.record)-rise_idx)))
                # After fit subtraction, area is a good indicator of whether a bad fit has badly messed up the remaining pulse
                corrected_pulse_area = np.sum(fitted_pulse)

                self.remaining_pulse_area.append(corrected_pulse_area)

                # NOTE tune this value
                # if corrected_pulse_area < -1000:
                #     print("Bad fit, remaining pulse mangled")

                # Get the timestamp, will be relative to the timestamp of the event at trigger point (first peak rise) in nanosecs
                self.true_timestamps.append(
                    self.timestamp + (rise_idx - self.rise_indices[0])*8)
                self.areas.append(np.sum(fitted_pulse))
                print("FIT AREA : ", np.sum(fitted_pulse))

                self.chi2.append(fit_function.GetChisquare())
                self.ndf.append(fit_function.GetNDF())

                # True count of rises in waveform, if fit has failed this should show how many we should have had
                self.pulse_count.append((len(self.rise_indices)))

                self.par0.append(fit_function.GetParameter(0))
                self.par1.append(fit_function.GetParameter(1))
                self.par2.append(fit_function.GetParameter(2))
                self.par3.append(fit_function.GetParameter(3))

                # Allow for potential cutting of waveforms near the end of the pulse - bad fits?
                self.rise_point.append(rise_idx)

                self.record_id.append(self.event_id)

                self.time_to_fit.append(fit_time)

                # plt.close()

                # plt.plot(self.record, label='Original Pulse')

                # Correct wfm
                self.record[rise_idx:] = self.record[rise_idx:] - fitted_pulse

                # print("Pulses in record: ", len(self.rise_indices))
                # print("Timestamps: ", self.true_timestamps)

                # plt.plot(fitted_pulse, label='Fitted pulse')
                # plt.plot(self.record[rise_idx:],
                #          label='Waveform minus Fitted Pulse')

                # plt.legend()
                # plt.show()

                del fitted_pulse
                del fit_function
                del graph
                # del canvas

            # idx is position in rise_indices, rise_idx is actual index of rise within the waveform self.record
            for idx, rise_idx in enumerate(self.rise_indices):

                # For case of multiple pulses, fit between rises
                try:
                    perform_fit(rise_idx, self.rise_indices[idx+1], idx)

                # For last pulse, or single pulse, fit from rise to end of record TODO Maybe we need to extend this for pulses close to the end of the record?
                except IndexError:
                    perform_fit(rise_idx, len(self.record), idx)

            # # Go through all the rise points, and fit between that and the next rise
            # for idx, rise_idx in enumerate(self.rise_indices, start=0):
            #     print(rise_idx)
            #     print(self.record[rise_idx])
            #     # plt.close()
            #     # plt.plot(self.record[rise_idx:])
            #     # plt.show()
            #     graph = ROOT.TGraph(len(self.record[rise_idx:]), np.linspace(
            #         rise_idx, len(self.record), len(self.record)-rise_idx), self.record[rise_idx:])

            #     # print("End point")
            #     # print(self.rise_indices[idx+1])

            #     # TODO add check for validity of
            #     try:
            #         fit_function = ROOT.TF1(
            #             'guo_fit', guo_fit, rise_idx, self.rise_indices[idx+1], 4)

            #         # These guesses seem to work well
            #         fit_function.SetParameters(
            #             1.5*self.rise_amplitudes[idx], self.rise_indices[idx], 50, 20)

            #         # Param constraints - needs work

            #         # fit_function.SetParLimits(0, 500, 3*self.rise_amplitudes[idx])
            #         # fit_function.SetParLimits(1, self.rise_indices[idx]-5, self.rise_indices[idx]+5)
            #         # fit_function.SetParLimits(2, 10, 100)
            #         # fit_function.SetParLimits(3, 3, 15)

            #         fit_result = graph.Fit(fit_function, "RSME")

            #         canvas = ROOT.TCanvas("canvas", "Guo Fit", 800, 600)
            #         graph.Draw("AP")
            #         fit_function.Draw("same")
            #         canvas.Update()
            #         canvas.Draw()

            #         # if fit_result.IsValid() == False:

            #         #     #print(self.rise_indices)
            #         #     print("Fit failed, moving to next pulse")
            #         #     input("Waiting")

            #         #     break

            #         input("Fit might have worked")
            #         # TODO Make this line less bad
            #         fitted_pulse = np.array([guo_fit([t], [fit_function.GetParameter(i) for i in range(
            #             fit_function.GetNpar())]) for t in np.linspace(rise_idx, len(self.record), len(self.record[rise_idx:]))])
            #         print("MEDIAN:", np.median(fitted_pulse))
            #         print("MEAN:", np.mean(fitted_pulse))

            #         # if np.median(fitted_pulse)>140:
            #         #     break

            #         # Get the timestamp, will be relative to the timestamp of the event at trigger point (first peak rise)
            #         self.true_timestamps.append(
            #             self.timestamp + (rise_idx - self.rise_indices[0])*8)
            #         self.areas.append(np.sum(fitted_pulse))
            #         self.chi2.append(fit_function.GetChisquare())
            #         self.ndf.append(fit_function.GetNDF())

            #         print("TT:", self.true_timestamps)
            #         print("AREAS: ", self.areas)
            #         print("CHI2:", self.chi2)
            #         print("NDF:", self.ndf)

            #         # Correct wfm
            #         self.record[rise_idx:] = self.record[rise_idx:] - \
            #             fitted_pulse
            #         # NOTE Plotting for fit checks
            #         plt.close()
            #         plt.plot(fitted_pulse, label='Fitted pulse')
            #         plt.plot(self.record[rise_idx:],
            #                  label='Waveform minus Fitted Pulse')

            #         plt.legend()
            #         plt.show()
            #         # plt.close()
            #         # if fit_result.IsValid() == False:
            #         #     input("Waiting")

            #     except IndexError:
            #         fit_function = ROOT.TF1(
            #             'guo_fit', guo_fit, rise_idx, len(self.record), 4)

            #         # These guesses arbitrarily work well
            #         fit_function.SetParameters(
            #             1.5*self.rise_amplitudes[idx], self.rise_indices[idx], 50, 20)
            #         # fit_function.SetParLimits(3, 3, 15)
            #         fit_result = graph.Fit(fit_function, "RSME")

            #         canvas = ROOT.TCanvas("canvas", "Guo Fit", 800, 600)
            #         graph.Draw("AP")
            #         fit_function.Draw("same")
            #         canvas.Update()
            #         canvas.Draw()

            #         # if fit_result.IsValid() == False:
            #         #     #print(self.rise_indices)
            #         #     print("Fit failed, moving to next pulse")
            #         #     input("Waiting")
            #         #     break

            #         input("Fit might have worked")
            #         # TODO this is overly simplified, use CHI2 and Ndf to determine quality of fit, save all, and then can filter
            #         # in post plotting

            #         # TODO Make this line less bad
            #         fitted_pulse = np.array([guo_fit([t], [fit_function.GetParameter(i) for i in range(
            #             fit_function.GetNpar())]) for t in np.linspace(rise_idx, len(self.record), len(self.record[rise_idx:]))])
            #         print("MEDIAN:", np.median(fitted_pulse))
            #         print("MEAN:", np.mean(fitted_pulse))

            #         # if np.median(fitted_pulse)>140:
            #         #     break

            #         # Get the timestamp, will be relative to the timestamp of the event at trigger point (first peak rise)
            #         self.true_timestamps.append(
            #             self.timestamp + (rise_idx - self.rise_indices[0])*8)
            #         self.areas.append(np.sum(fitted_pulse))
            #         print("TT:", self.true_timestamps)
            #         print("AREAS: ", self.areas)
            #         self.chi2.append(fit_function.GetChisquare())
            #         self.ndf.append(fit_function.GetNDF())
            #         print("CHI2:", self.chi2)
            #         print("NDF:", self.ndf)
            #         # This is an extracted pulse, so get the information
            #         # areas.append(np.sum(fitted_pulse))
            #         # timestamps.append(rise_idx*)

            #         # Correct wfm
            #         self.record[rise_idx:] = self.record[rise_idx:] - \
            #             fitted_pulse
            #         # NOTE Plotting for fit checks
            #         plt.close()
            #         plt.plot(fitted_pulse, label='Fitted pulse')
            #         plt.plot(self.record[rise_idx:],
            #                  label='Waveform minus Fitted Pulse')

            #         plt.legend()
            #         plt.show()
            #         # plt.close()
            #         # plt.plot(fitted_pulse, label='Fit')
            #         # plt.plot(self.record[rise_idx:] - fitted_pulse, label='Subtracted')
            #         # plt.legend()
            #         # plt.show()
            #         # plt.close()

            #         # plt.close()

            #     # input("Continue...")

                # del fitted_pulse
                # del fit_function
                # del graph
                # del canvas

        # else:
        #     self.true_timestamps.append(self.timestamp)
        #     self.areas.append(np.sum(self.record[self.rise_indices[0]:]))
        #     print("TT:", self.true_timestamps)
        #     print("AREAS: ", self.areas)
        #     self.chi2.append(0)
        #     self.ndf.append(0)
        #     print("CHI2:", self.chi2)
        #     print("NDF:", self.ndf)

            # Create a TF1 object for the fit function
            # fit_function = ROOT.TF1("landau", 'landau', self.peak_times[0]-prefit_window, self.peak_times[0]+postfit_window)
            # fit_function = ROOT.TF1('guo_fit', guo_fit, self.peak_times[0]-prefit_window, self.peak_times[0]+postfit_window, 4)
            # fit_function.SetParameters(2*self.peak_heights[0], self.peak_times[0], 50, 20)

            # #Perform the fit
            # fit_result = graph.Fit(fit_function, "RS")
            # print(fit_result.IsValid())
            # if fit_result.IsValid() ==False:
            #     print("FIT FAILURE")

            #     canvas = ROOT.TCanvas("canvas", "Guo Fit", 800, 600)
            #     graph.Draw("AP")

            #     fit_function.Draw("same")

            #     canvas.Update()
            #     canvas.Draw()

            #     # check if at end of pulse and fitting not possible in some way, if this is the case,
            #     # probably not necessary to throw away whole pulse

            #     input("Press Enter to close the canvas...")
            # else:
            #     print("FIT SUCCESSFUL")

    def fit(self, prefit_window=200, postfit_window=500):

        if len(self.peak_times) >= 1:

            # # Convert numpy arrays to PyROOT arrays
            x_array = self.time[self.time > self.peak_times[0]-prefit_window]
            y_array = self.record[self.time > self.peak_times[0]-prefit_window]

            # Create a TGraph object
            graph = ROOT.TGraph(
                len(self.time[self.time > self.peak_times[0]-prefit_window]), x_array, y_array)

            def guo_fit(x, par):
                '''par[0] - A
                   par[1] - t0
                   par[2] - theta1
                   par[3] - theta2'''
                return par[0]*(np.exp(-(x[0]-par[1])/par[2]) - np.exp(-(x[0]-par[1])/par[3]))

            # Create a TF1 object for the fit function
            # fit_function = ROOT.TF1("landau", 'landau', self.peak_times[0]-prefit_window, self.peak_times[0]+postfit_window)
            fit_function = ROOT.TF1(
                'guo_fit', guo_fit, self.peak_times[0]-prefit_window, self.peak_times[0]+postfit_window, 4)
            fit_function.SetParameters(
                2*self.peak_heights[0], self.peak_times[0], 50, 20)

            # Perform the fit
            fit_result = graph.Fit(fit_function, "RS")
            print(fit_result.IsValid())
            if fit_result.IsValid() == False:
                print("FIT FAILURE")

                canvas = ROOT.TCanvas("canvas", "Guo Fit", 800, 600)
                graph.Draw("AP")

                fit_function.Draw("same")

                canvas.Update()
                canvas.Draw()

                # check if at end of pulse and fitting not possible in some way, if this is the case,
                # probably not necessary to throw away whole pulse

                input("Press Enter to close the canvas...")
            else:
                print("FIT SUCCESSFUL")

            # fitted_pulse = np.array([fit_function.GetParameter(0) * ROOT.TMath.Landau(t, fit_function.GetParameter(1), fit_function.GetParameter(2)) for t in self.time[self.time>self.peak_times[0]-prefit_window]])
            fitted_pulse = np.array([guo_fit([t], [fit_function.GetParameter(i) for i in range(
                fit_function.GetNpar())]) for t in self.time[self.time > self.peak_times[0]-prefit_window]])

            # Added cut for above 0
            # area = np.sum(fitted_pulse[fitted_pulse>0])
            area = np.sum(fitted_pulse)
            self.area = area

            # EDIT THIS TO CORRECT THE TIMESTAMP
            self.timestamp = self.timestamp
            print("AREA: ", area)

            plt.close()
            plt.plot(self.time, self.record, label='Original full pulse')
            self.record[self.time > self.peak_times[0]-prefit_window] = self.record[self.time >
                                                                                    self.peak_times[0]-prefit_window] - fitted_pulse
            plt.plot(self.time[self.time > self.peak_times[0]], self.record[self.time >
                     self.peak_times[0]], label='Original pulse - fit region', linestyle='-')
            plt.plot(self.time[self.time > self.peak_times[0] -
                     prefit_window], fitted_pulse, label='Fit pulse')
            plt.plot(self.time, self.record, label='Subtracted fit pulse')

            plt.legend()
            plt.show()

            plt.close()

            # Arbitrary offset of 50 now before another peak is allowed
            try:
                self.get_peaks(
                    minimum_time=self.peak_times[0]+50, threshold=500)
            except:
                self.peak_heights = []
                self.peak_times = []
                print("No peaks found")

            # Now find area of this fitted exponential from the maximum onwards

            # Then look at doing 2 pulse situations where we need to start
            # iterative subtraction

            # Fit to first, find area, subtract from full waveform
            # Fit to next, find area, subtract from full waveform
            # Fit to next ... etc.
            # In the end the 'waveform' should be flat and we have extracted
            # N areas where N is number of peaks found
        else:
            print("Insufficient peaks")
