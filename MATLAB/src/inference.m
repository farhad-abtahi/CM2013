function predictions = inference_make_inference(model, holdout_data)
%% Makes predictions on the hold-out data using the trained model.

fprintf('Making inference on hold-out data...\n');
predictions = predict(model, holdout_data);
end

function inference_generate_submission_file(predictions, record_numbers, epoch_numbers, config)
%% Generates a submission CSV file.

fprintf('Generating submission file: %s...\n', config.SUBMISSION_FILE);

% Create a table for the submission file
submissionTable = table(record_numbers(:), epoch_numbers(:), predictions(:), ...
    'VariableNames', {'record_number', 'epoch_number', 'label'});

% Define the full path for the submission file
submissionFilePath = fullfile(config.DATA_DIR, config.SUBMISSION_FILE);

% Write the table to a CSV file
writetable(submissionTable, submissionFilePath);

fprintf('Submission file generated successfully.\n');

end
