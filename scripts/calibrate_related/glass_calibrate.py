import time
from tobiiglassesctrl.controller import TobiiGlassesController


ipv4_address = "192.168.71.50"
def main():

	tobiiglasses = TobiiGlassesController(ipv4_address)
	print(tobiiglasses.get_battery_info())
	print(tobiiglasses.get_storage_info())

	if tobiiglasses.is_recording():
		rec_id = tobiiglasses.get_current_recording_id()
		tobiiglasses.stop_recording(rec_id)

	project_name = "pros_calibrate2"
	project_id = tobiiglasses.create_project(project_name)

	participant_name = "wyx"
	participant_id = tobiiglasses.create_participant(project_id, participant_name)

	calibration_id = tobiiglasses.create_calibration(project_id, participant_id)
	input("Put the calibration marker in front of the user, then press enter to calibrate")
	tobiiglasses.start_calibration(calibration_id)

	res = tobiiglasses.wait_until_calibration_is_done(calibration_id)

	if res is False:
		print("Calibration failed!")
		exit(1)

if __name__ == '__main__':
    main()