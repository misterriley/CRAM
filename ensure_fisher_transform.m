function m = ensure_fisher_transform(m, to_z)
    test_val = m(1,1,1);
    if test_val ~= 1 && abs(tanh(test_val) - 1) > .01
        disp("Expecting 1 or arctan(~.999) in location [1,1,1]. Value is %f. Exiting.\n", test_val);
        exit;
    end

    if ~to_z && test_val ~= 1
        % Apply Fisher inverse transformation to data if needed
        m = tanh(m);
        for i = 1:size(m,1)
            m(i,i,:) = 1;
        end
    end

    if to_z && test_val == 1
        for i = 1:size(m,1)
            m(i,i,:) = .99999;
        end
        m = atanh(m);
    end
end