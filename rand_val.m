function val = rand_val(mu, sigma, max)
    val = round(mu + sigma * randn());
    if (val > max)
        d = val - max;
        val = val - 2*d;
    end
end