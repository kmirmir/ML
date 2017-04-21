package kr.ac.jejunu;

import javax.sql.DataSource;
import java.sql.*;

public class ProductDao {


    JdbcContext jdbcContext;

    public ProductDao() {

    }

    public void setJdbcContext(JdbcContext jdbcContext) {
        this.jdbcContext = jdbcContext;
    }



    public Product get(Long id) throws ClassNotFoundException, SQLException {
        StatementStrategy statementStrategy = connection -> {
            PreparedStatement preparedStatement;
            preparedStatement = connection.prepareStatement("select * from product where id = ?");
            preparedStatement.setLong(1, id);
            return preparedStatement;
        };
        return jdbcContext.JdbcContextWithStatementForQuery(statementStrategy);
    }


    public void add(Product product) throws ClassNotFoundException, SQLException {
        String sql = "insert into product (id, title, price) VALUES (?,?,?)";
        Object[] params = new Object[]{product.getId(), product.getTitle(), product.getPrice()};

        jdbcContext.update(sql, params);
    }



    public void update(Product product) throws ClassNotFoundException, SQLException {
        String sql = "update product set title = ?, price = ? where id = ?";
        Object[] params = new Object[]{product.getTitle(), product.getPrice(), product.getId()};

        jdbcContext.update(sql, params);
    }

    public void delete(Long id)  throws ClassNotFoundException, SQLException {
        String sql = "delete from product where id = ?";
        Object[] params = new Object[]{id};
        jdbcContext.update(sql, params);
    }


}
