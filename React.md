# Fundamental

React is a View library reacts to state change and sync to DOM

npm i -g create-react-app

create-react-app react-app

```shell
npx create-react-app my-app
cd my-app
npm start
```

render() return JSX

Babel: modern js compiler: translate JSX to react element



# Reusable Components

bootstrap@4.1.1

Create components folder and xxx.jsx

<React.Fragment>

JSX elemetns is an object of js

{this.state.tags.length === 0 && "Create a new tag"}



```
  handleIncrement = (product) => {
    console.log(product);
    this.setState({ count: this.state.count + 1 });
  };
  
  onClick={() => this.handleIncrement(product)}
```



```
<ul>
        {this.state.tags.map((tag) => (
          <li key={tag}>{tag}</li>
        ))}
</ul>
```



**Pass Data**

Prop is read-only

```
{this.state.counters.map((counter) => (
<Counter key={counter.id} value={counter.value}>
	<h4>Counter #{counter.id}</h4>
</Counter>
))}

state = {value: this.props.value}
{this.props.children}
```



**Raise and handle events**

```jsx
Counter:
<button
	onClick={() => this.props.onDelete(this.props.counter.id)}
>

Counters:
handleDelete = (counterId) => {
const counters = this.state.counters.filter((c) => c.id !== counterId);
this.setState({ counters: counters });
};
  
<Counter
	onDelete={this.handleDelete}
	key={counter.id}
	counter={counter}
>
```



controlled components don't have own local state 

only receive data via props and raise events when data need to change

```jsx
  handleIncrement = (counter) => {
    const counters = [...this.state.counters];
    const index = counters.indexOf(counter);
    counters[index] = { ...counter };
    counters[index].value++;
    this.setState({ counters });
  };
```



**Lifting State Up** 

Multiple components in sync



**Stateless Functional Component**

use a function

Short cut: sfc, imfc, cc

```jsx
const NavBar = (props) => {
  return (
    <nav class="navbar navbar-light bg-light">
      <a class="navbar-brand" href="#">
        Navbar{" "}
        <span className="badge badge-pill badge-secondary">
          {props.totalCounters}
        </span>
      </a>
    </nav>
  );
};

export default NavBar;
```



**object destructing**

```jsx
const NavBar = ({ totalCounters }) => {}

const { counters, onReset, onDelete, onIncrement } = this.props;
```



**Lifecycle Hooks**

can only used in component class

Mount: constructor -> render -> componentDidMount

Update: render -> componentDidUpdate

Unmount: componentWillUnmount











# Debug



 