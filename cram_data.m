classdef cram_data < handle
    properties
        best_cod
        best_dim
        best_r
        best_rho
        best_rmse
        cv_folds
        cv_repeats
        dataset
        distmatrix
        extract_networks
        folder
        hcp_outcome % in the hcp data set we have multiple outcome measures
        limit
        mats
        mtrc
        neg_net
        outcomes
        p_thresh
        p_regression
        perm_repeats
        perm_rs
        pos_net
        sig_pos_edges
        sig_neg_edges
        target
        total_time
        use_fdr
        wass_dim
    end

    properties (Constant)
        file_name = "cram_data.mat";
    end

    methods
        function obj = cram_data(folder)
            obj.folder = folder;
        end

        function save(obj)
            fprintf("Saving %s....\n", cram_data.file_name);
            save_file = fullfile(obj.folder, cram_data.file_name);
            save(save_file, "obj")
            fprintf("\bdone.\n");
        end
    end

    methods (Static)
        function obj = load(folder)
            save_file = fullfile(folder, cram_data.file_name);
            if isfile(save_file)
                load(save_file, "obj");
            else
                obj = cram_data(folder);
            end
        end

        function ex = check_for_variable(obj, var, print_status_if_exists)
            ex = false;
            if isfield(obj, var) || isprop(obj, var)
                eval_string = sprintf("x = obj.%s;", var);
                eval(eval_string);
                if ~isempty(x) && ~all(all(all(isnan(x))))
                    ex = true;
                end
            end

            if ex && print_status_if_exists
                fprintf("%s already exists at %s and will be used in this CRAM session.\n", var, obj.folder);
                fprintf("To recompute, please move, edit, or rename cram_data.m in this folder.\n");
            end
        end
    end
end